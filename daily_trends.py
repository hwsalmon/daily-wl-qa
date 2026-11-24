"""Trend analysis of the Daily QA history: what to look at, how it is judged, and the figures.

Four things are followed for each machine:
  1. radiation isocenter: the maximum displacement from the isocenter (walk radius) and its DRIFT - a straight-line fit over time (mm per 30 days, with its p-value) and the change of the
     recent median from the baseline (the first sessions), plus the raw SI / lateral offsets at each gantry angle that move when the gantry, the imager or the beam shifts;
  2. jaw size and 3. MLC size: the measured field size minus the reference, with the same drift statistics and the warning / fail limits;
  4. picket fence: the largest and RMS leaf error, the worst leaf of every session, and whether it is the SAME leaf again and again (a real, persistent leaf offset) or random.
No Qt here: `build(db_path, machine, key, refs, limits)` returns the figure (PNG bytes), a summary text and the table rows, so it can be tested without a display.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import sqlite3
from collections import Counter

import numpy as np

KEYS = (("walk", "Radiation isocenter - radius and drift"),
        ("jaw", "Field size - jaw"),
        ("mlc", "Field size - MLC"),
        ("pf", "Picket fence - max / RMS / worst leaf"),
        ("leaf", "Picket fence - is the same leaf always the worst?"))
GRN, AMB, RED = "#66bb6a", "#ffa726", "#ef5350"
MIN_FOR_TREND = 6          # sessions before a drift is quoted
SIG_P = 0.01               # a slope is called a drift when it is this unlikely to be noise AND moves the value by more than MIN_DRIFT
MIN_DRIFT = {"walk": 0.10, "size": 0.15, "pf": 0.05}      # mm over the 90 days of the fit, below which a statistically firm slope is still not worth a flag


def _dt(date: str, time: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    except ValueError:
        return dt.datetime.strptime(date, "%Y-%m-%d")


def load(db_path, machine: str) -> dict:
    """{'wl': [dict], 'pf': [dict]} for one machine, oldest first. Missing tables / columns give empty lists."""
    out = {"wl": [], "pf": []}
    try:
        c = sqlite3.connect(str(db_path)); c.row_factory = sqlite3.Row
    except sqlite3.Error:
        return out
    try:
        for r in c.execute("SELECT * FROM wl_records WHERE machine_name=? ORDER BY date, time", (machine,)):
            d = dict(r); d["when"] = _dt(d["date"], d["time"]); out["wl"].append(d)
    except sqlite3.Error:
        pass
    try:
        for r in c.execute("SELECT * FROM pf_records WHERE machine_name=? ORDER BY date, time", (machine,)):
            d = dict(r); d["when"] = _dt(d["date"], d["time"])
            try:
                d["devs"] = np.asarray(json.loads(d.get("leaf_dev_json") or "[]"), float)
                d["ys"] = np.asarray(json.loads(d.get("leaf_y_json") or "[]"), float)
            except ValueError:
                d["devs"], d["ys"] = np.array([]), np.array([])
            d["tilt"], d["resid"] = tilt_and_residual(d["devs"], d["ys"])
            out["pf"].append(d)
    except sqlite3.Error:
        pass
    c.close()
    return out


def tilt_and_residual(devs, ys):
    """The straight-line tilt of the leaf errors along y (mm change from one end of the picket to the other) and the errors with that line removed. A tilt is not a leaf problem: it is the same
    at every leaf position and comes from a rotation / scale of the MLC against the imager or the phantom, so it is tracked on its own and taken out before looking for a bad leaf."""
    devs = np.asarray(devs, float); ys = np.asarray(ys, float)
    if len(devs) < 8 or len(ys) != len(devs) or np.ptp(ys) <= 0:
        return None, devs
    b, a = np.polyfit(ys, devs, 1)
    return float(b * np.ptp(ys)), devs - (a + b * ys)


def machines_with_data(db_path) -> list[str]:
    names = set()
    try:
        c = sqlite3.connect(str(db_path))
        for tbl in ("wl_records", "pf_records"):
            try:
                names |= {r[0] for r in c.execute(f"SELECT DISTINCT machine_name FROM {tbl}")}
            except sqlite3.Error:
                pass
        c.close()
    except sqlite3.Error:
        pass
    return sorted(names)


# ── statistics ────────────────────────────────────────────────────────────────────────────────────────────────────────

def drift(times, y, kind: str = "walk") -> dict:
    """Straight-line drift of y over time: mm per 30 days, p-value, the change over 90 days, baseline (median of the first third, at most 10) and recent (last 10, at most a third) medians."""
    y = np.asarray(y, float); ok = np.isfinite(y)
    t = np.array([(x - times[0]).total_seconds() / 86400.0 for x in times])[ok]; y = y[ok]
    n = len(y)
    res = {"n": n, "slope_30d": None, "p": None, "change_90d": None, "baseline": None, "recent": None, "shift": None, "flag": False, "text": ""}
    if n == 0:
        return res
    k = max(1, min(10, n // 3))
    res["baseline"], res["recent"] = float(np.median(y[:k])), float(np.median(y[-k:]))
    res["shift"] = res["recent"] - res["baseline"]
    if n < MIN_FOR_TREND or np.ptp(t) < 7.0:
        res["text"] = f"{n} session(s): too few for a drift estimate (need {MIN_FOR_TREND} over at least a week)."
        return res
    from scipy.stats import linregress
    lr = linregress(t, y)
    res["slope_30d"], res["p"], res["change_90d"] = float(lr.slope * 30.0), float(lr.pvalue), float(lr.slope * 90.0)
    lim = MIN_DRIFT.get(kind, 0.1)
    res["flag"] = bool(lr.pvalue < SIG_P and abs(res["change_90d"]) >= lim)
    res["text"] = (f"{n} sessions {times[0]:%Y-%m-%d} to {times[-1]:%Y-%m-%d}: slope {res['slope_30d']:+.3f} mm per 30 days (p = {res['p']:.3g}, {res['change_90d']:+.2f} mm per 90 days); "
                   f"median of the first {k} = {res['baseline']:.3f}, of the last {k} = {res['recent']:.3f} ({res['shift']:+.3f}).  "
                   + ("DRIFT: statistically firm and larger than " + f"{lim:.2f} mm per 90 days." if res["flag"] else "No drift above the noise."))
    return res


MIN_STEP = {"walk": 0.15, "size": 0.30, "pf": 0.10}


def find_steps(times, y, kind: str = "walk", max_steps: int = 3):
    """Step changes (a calibration, a repair, a different setup): binary segmentation on the medians. A split is a step when the median of up to 5 sessions after it differs from the median of
    up to 5 before it by more than max(MIN_STEP, 4 x the session-to-session noise); noise is a robust estimate from consecutive differences. Returns [(index of the first session after the
    step, shift, date before, date after)], largest first."""
    y = np.asarray(y, float); n = len(y)
    if n < 8:
        return []
    ok = np.isfinite(y)
    if ok.sum() < 8:
        return []
    sigma = 1.4826 * np.median(np.abs(np.diff(y[ok]) - np.median(np.diff(y[ok])))) / np.sqrt(2.0) if ok.sum() > 3 else 0.0
    need = max(MIN_STEP.get(kind, 0.15), 4.0 * sigma)

    def best(lo, hi):
        bk, bs = None, 0.0
        for k in range(lo + 3, hi - 2):
            a, b = y[max(lo, k - 5):k], y[k:min(hi, k + 5)]
            sh = float(np.median(b) - np.median(a))
            if abs(sh) > abs(bs):
                bk, bs = k, sh
        if bk is not None:                                           # place the step where a two-level fit of the neighbourhood is best (the windows alone cannot tell k from k-1)
            w0, w1 = max(lo, bk - 8), min(hi, bk + 8)
            def sse(k):
                a, b = y[w0:k], y[k:w1]
                return float(np.sum((a - a.mean()) ** 2) + np.sum((b - b.mean()) ** 2)) if len(a) >= 2 and len(b) >= 2 else np.inf
            bk = min(range(max(lo + 3, bk - 3), min(hi - 2, bk + 4)), key=sse)
            bs = float(np.median(y[bk:min(hi, bk + 5)]) - np.median(y[max(lo, bk - 5):bk]))
        return bk, bs
    found, segs = [], [(0, n)]
    while segs and len(found) < max_steps:
        cand = [(best(lo, hi), lo, hi) for lo, hi in segs if hi - lo >= 6]
        cand = [c for c in cand if c[0][0] is not None and abs(c[0][1]) >= need]
        if not cand:
            break
        (k, sh), lo, hi = max(cand, key=lambda c: abs(c[0][1]))
        found.append((k, sh, times[k - 1], times[k]))
        segs.remove((lo, hi)); segs += [(lo, k), (k, hi)]
    return sorted(found, key=lambda f: -abs(f[1]))


def steps_text(steps) -> str:
    if not steps:
        return ""
    return "STEP CHANGE: " + "; ".join(f"{sh:+.2f} mm between {a:%Y-%m-%d} and {b:%Y-%m-%d}" for _k, sh, a, b in steps) + ". "


def mark_steps(ax, steps):
    for _k, sh, a, b in steps:
        ax.axvline(a + (b - a) / 2, color="#ff80ab", ls=":", lw=1.6)
        ax.annotate(f"{sh:+.2f}", (a + (b - a) / 2, ax.get_ylim()[1]), xytext=(3, -12), textcoords="offset points", color="#ff80ab", fontsize=9)


def moving_average(y, k: int = 7):
    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    k = max(2, min(k, len(y)))
    pad = np.r_[np.full(k - 1, y[0]), y]
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def leaf_consistency(pf_rows: list[dict]) -> dict:
    """Is the same leaf the worst again and again? worst-leaf counts (with a binomial p-value for the commonest leaf against 'any leaf equally likely', corrected for having picked the commonest of the L leaves) and, per leaf, the mean signed
    deviation over all sessions with its standard error: a leaf whose mean is several standard errors from zero is persistently displaced, whichever session it was the worst in."""
    rows = [r for r in pf_rows if len(r["devs"]) >= 8]
    if not rows:
        return {"n": 0}
    L = Counter(len(r["devs"]) for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if len(r["devs"]) == L]
    D = np.vstack([r["resid"] for r in rows])                 # tilt removed
    Draw = np.vstack([r["devs"] for r in rows])
    tilts = np.array([r["tilt"] if r["tilt"] is not None else np.nan for r in rows])
    worst = [int(np.argmax(np.abs(r))) for r in D]
    cnt = Counter(worst)
    n = len(rows)
    top, top_n = cnt.most_common(1)[0]
    from scipy.stats import binomtest
    p = min(1.0, L * float(binomtest(top_n, n, 1.0 / L, alternative="greater").pvalue)) if n >= 5 else None      # the commonest of L leaves was picked AFTER looking: correct for the L comparisons
    mean = D.mean(axis=0)
    sem = D.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(L, np.nan)
    z = np.where(sem > 0, mean / np.where(sem > 0, sem, 1), 0.0)
    persistent = [(int(i), float(mean[i]), float(sem[i]), float(z[i])) for i in np.argsort(-np.abs(z)) if abs(z[i]) >= 4.0 and abs(mean[i]) >= 0.05][:6]
    ys = rows[-1]["ys"] if len(rows[-1]["ys"]) == L else np.arange(L, dtype=float)
    adj = cnt[top] + cnt.get(top - 1, 0) + cnt.get(top + 1, 0)
    raw_worst = Counter(int(np.argmax(np.abs(r))) for r in Draw)
    return {"n": n, "L": L, "D": D, "Draw": Draw, "tilts": tilts, "raw_top": raw_worst.most_common(1)[0], "raw_ends": sum(v for k, v in raw_worst.items() if k < 2 or k >= L - 2), "worst": worst, "counts": cnt, "top": top, "top_n": top_n, "top_frac": top_n / n, "expected": 1.0 / L, "p": p, "mean": mean, "sem": sem,
            "z": z, "persistent": persistent, "ys": ys, "near_top": adj, "times": [r["when"] for r in rows]}


def leaf_text(c: dict) -> str:
    if not c.get("n"):
        return "No picket-fence records for this machine yet."
    ys = c["ys"]; lab = lambda i: f"leaf {i} (y = {ys[i]:+.1f} mm)"
    s = ""
    tl = c["tilts"][np.isfinite(c["tilts"])]
    if len(tl):
        s += (f"Tilt along the picket: the leaf errors change by {np.mean(tl):+.2f} mm from one end to the other (session-to-session SD {np.std(tl, ddof=1) if len(tl) > 1 else 0:.2f} mm). "
              f"A tilt is a rotation / scale of the MLC against the imager or the phantom, not a bad leaf; it makes an END leaf the 'worst' ({c['raw_ends']} of {c['n']} sessions without correcting for it), so it is taken out below. ")
    s += f"With the tilt removed, {c['n']} sessions: the worst leaf was {lab(c['top'])} in {c['top_n']} of them ({100 * c['top_frac']:.0f} %; {100 * c['expected']:.1f} % expected if every leaf were equally likely"
    s += f", p = {c['p']:.2g}" if c["p"] is not None else ""
    s += "). "
    if c["p"] is not None and c["p"] < 0.01 and c["top_n"] >= 3:
        s += "THE SAME LEAF IS REPEATEDLY THE WORST - a persistent leaf offset, worth a leaf calibration check. "
    else:
        s += "The worst leaf moves around (no repeating leaf). "
    if c["persistent"]:
        s += "Persistently displaced (mean over all sessions, in standard errors): " + "; ".join(f"{lab(i)} {m:+.2f} mm ({z:+.0f} SE)" for i, m, _e, z in c["persistent"]) + "."
    else:
        s += "No leaf is persistently displaced."
    return s


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def _style(fig, ax):
    fig.patch.set_facecolor("#111111")
    for a in np.atleast_1d(ax):
        a.set_facecolor("#1e1e1e"); a.tick_params(colors="white", labelsize=8)
        for sp in a.spines.values():
            sp.set_edgecolor("#555555")
        a.xaxis.label.set_color("white"); a.yaxis.label.set_color("white"); a.title.set_color("white")
        a.grid(alpha=0.2)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=110, bbox_inches="tight", facecolor="#111111")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def _legend(ax, **k):
    ax.legend(facecolor="#2b2b2b", edgecolor="#555555", labelcolor="white", fontsize=8, **k)


def _dates(ax, fig):
    import matplotlib.dates as md
    ax.xaxis.set_major_formatter(md.ConciseDateFormatter(md.AutoDateLocator()))


def gapped(times, y, days: float = 21.0):
    """times / y with a NaN inserted wherever the history has a gap of more than `days`, so a line is not drawn across a period with no data."""
    t2, y2 = [], []
    for i, (a, b) in enumerate(zip(times, y)):
        if i and (a - times[i - 1]).days > days:
            t2.append(times[i - 1] + (a - times[i - 1]) / 2); y2.append(np.nan)
        t2.append(a); y2.append(b)
    return t2, np.array(y2, float)


def _trend_line(ax, times, y, d):
    if d["slope_30d"] is None:
        return
    t = np.array([md_num(x) for x in times]); y = np.asarray(y, float)
    ok = np.isfinite(y)
    sl, ic = np.polyfit(t[ok] - t[0], y[ok], 1)
    ax.plot([times[0], times[-1]], [ic, ic + sl * (t[-1] - t[0])], "--", color="#ff80ab" if d["flag"] else "#bbbbbb", lw=1.4, label=f"trend {d['slope_30d']:+.3f} mm / 30 d" + ("  DRIFT" if d["flag"] else ""))


def md_num(x):
    import matplotlib.dates as md
    return float(md.date2num(x))


def fig_walk(wl, tol):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    _style(fig, [a1, a2])
    t = [r["when"] for r in wl]; y = [r["walk_circle_r"] for r in wl]
    d = drift(t, y, "walk")
    a1.axhline(tol, color=RED, ls="--", lw=1.4, label=f"limit {tol:.1f} mm (SRS)")
    a1.plot(*gapped(t, y), "-", color="#64b5f6", lw=1, alpha=0.7)
    a1.scatter(t, y, c=[GRN if v <= 0.9 * tol else AMB if v <= tol else RED for v in y], s=34, edgecolors="#64b5f6", linewidths=0.8, zorder=4)
    if len(y) >= 5:
        a1.plot(t, moving_average(y, 7), color="#fff176", lw=2, label="7-session average")
    _trend_line(a1, t, y, d)
    a1.set_ylim(0, max(tol * 1.15, max(y) * 1.2)); a1.set_ylabel("Max displacement from isocenter (mm)")
    d["steps"] = find_steps(t, y, "walk"); mark_steps(a1, d["steps"])
    a1.set_title("Radiation isocenter radius (maximum displacement of any beam from the isocenter)", fontsize=11, fontweight="bold")
    _legend(a1, loc="upper left")
    for ang, col in ((0, "#64b5f6"), (90, "#81c784"), (180, "#ffb74d"), (270, "#f06292")):
        v = [r.get(f"raw_dy_G{ang}") for r in wl]
        if any(x is not None for x in v):
            tg, vg = gapped(t, [np.nan if x is None else x for x in v])
            a2.plot(tg, vg, "o-", ms=3, lw=1, color=col, label=f"G{ang}")
    a2.set_ylabel("Raw SI offset: void minus field centre (mm)"); a2.axhline(0, color="#777", lw=0.8)
    a2.set_title("Where the offset sits at each gantry angle (a step or slope in one angle = a shift of that part of the machine, or of the phantom)", fontsize=9)
    _legend(a2, loc="upper left", ncol=4)
    _dates(a2, fig); fig.tight_layout()
    return _png(fig), d


def fig_size(wl, kind, ref, warn, fail, label):
    import matplotlib.pyplot as plt
    col = "avg_mlc_mm" if kind == "mlc" else "avg_jaw_mm"
    rows = [r for r in wl if r.get(col) is not None]
    t = [r["when"] for r in rows]; y = [r[col] - ref for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.2)); _style(fig, ax)
    d = drift(t, y, "size") if rows else drift([dt.datetime.now()], [])
    ax.axhline(0, color="#999", lw=1)
    for lv, c, nm in ((warn, AMB, "warn"), (fail, RED, "fail")):
        ax.axhline(lv, color=c, ls="--", lw=1.2, label=f"{nm} |Δ| > {lv:.1f} mm"); ax.axhline(-lv, color=c, ls="--", lw=1.2)
    if rows:
        ax.plot(*gapped(t, y), "-", color="#64b5f6", lw=1, alpha=0.7)
        ax.scatter(t, y, c=[GRN if abs(v) <= warn else AMB if abs(v) <= fail else RED for v in y], s=34, edgecolors="#64b5f6", linewidths=0.8, zorder=4)
        if len(y) >= 5:
            ax.plot(t, moving_average(y, 7), color="#fff176", lw=2, label="7-session average")
        _trend_line(ax, t, y, d)
        d["steps"] = find_steps(t, y, "size"); mark_steps(ax, d["steps"])
        sd = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
        ax.text(0.99, 0.03, f"mean {np.mean(y):+.3f} mm   scatter (SD) {sd:.3f} mm   range {min(y):+.2f} to {max(y):+.2f}", transform=ax.transAxes, ha="right", color="white", fontsize=9)
    ax.set_ylabel(f"{label} size minus reference (mm)"); ax.set_title(f"{label} field size: variation from the reference ({ref:.2f} mm)", fontsize=11, fontweight="bold")
    _legend(ax, loc="upper left"); _dates(ax, fig); fig.tight_layout()
    return _png(fig), d


def fig_pf(pf, warn, tol):
    import matplotlib.pyplot as plt
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 9.4), sharex=True, gridspec_kw={"height_ratios": [1.1, 0.8, 1]}); _style(fig, [a1, a2, a3])
    t = [r["when"] for r in pf]; mx = [r["max_dev_mm"] for r in pf]; rms = [r["rms_dev_mm"] for r in pf]
    tl = [np.nan if r["tilt"] is None else r["tilt"] for r in pf]
    d = drift(t, mx, "pf")
    a1.axhline(tol, color=RED, ls="--", lw=1.3, label=f"fail > {tol:.1f} mm"); a1.axhline(warn, color=AMB, ls="--", lw=1.2, label=f"warn ≥ {warn:.1f} mm")
    a1.plot(t, mx, "o-", color="#64b5f6", ms=4, lw=1, label="max |leaf error|")
    a1.plot(t, rms, "s-", color="#ce93d8", ms=3, lw=1, label="RMS leaf error")
    _trend_line(a1, t, mx, d)
    d["steps"] = find_steps(t, mx, "pf"); mark_steps(a1, d["steps"])
    a1.set_ylabel("mm"); a1.set_title("Picket fence: largest and RMS leaf error", fontsize=11, fontweight="bold"); _legend(a1, loc="upper left", ncol=2)
    a2.plot(t, tl, "o-", color="#80cbc4", ms=4, lw=1); a2.axhline(0, color="#777", lw=0.8)
    a2.set_ylabel("End-to-end tilt (mm)"); a2.set_title("Tilt of the leaf errors along the picket (systematic: MLC / imager / phantom rotation or scale - not a single leaf)", fontsize=9)
    a2 = a3
    ys = np.array([r["worst_leaf"] for r in pf], float); dev = np.array([r["worst_dev_mm"] for r in pf], float)
    sc = a2.scatter(t, ys, c=dev, cmap="coolwarm", vmin=-max(0.3, np.nanmax(np.abs(dev))), vmax=max(0.3, np.nanmax(np.abs(dev))), s=46, edgecolors="white", linewidths=0.5)
    cb = fig.colorbar(sc, ax=a2, pad=0.01); cb.set_label("worst leaf's signed error (mm)", color="white", fontsize=8); cb.ax.tick_params(colors="white", labelsize=7)
    a2.set_ylabel("Worst leaf (index along y)"); a2.set_title("Which leaf was the worst, each day, before taking the tilt out (a horizontal line of dots = the same leaf every time)", fontsize=9)
    _dates(a2, fig); fig.tight_layout()
    return _png(fig), d


def fig_leaf(c):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 8)); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[1.6, 1])
    ax_h, ax_c, ax_m = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]); _style(fig, [ax_h, ax_c, ax_m])
    D, ys, L = c["D"], c["ys"], c["L"]
    v = max(0.2, float(np.nanpercentile(np.abs(D), 98)))
    im = ax_h.imshow(D, aspect="auto", cmap="coolwarm", vmin=-v, vmax=v, extent=[-0.5, L - 0.5, len(D) - 0.5, -0.5], interpolation="nearest")
    cb = fig.colorbar(im, ax=ax_h, pad=0.01); cb.set_label("leaf error (mm)", color="white", fontsize=8); cb.ax.tick_params(colors="white", labelsize=7)
    ax_h.set_xlabel("Leaf index (along y)"); ax_h.set_ylabel("Session (oldest at top)")
    ax_h.set_title("Every leaf, every session, tilt removed - a vertical stripe is a leaf that is always off", fontsize=10, fontweight="bold")
    cnt = [c["counts"].get(i, 0) for i in range(L)]
    ax_c.bar(range(L), cnt, color=["#ef5350" if i == c["top"] else "#64b5f6" for i in range(L)])
    ax_c.axhline(c["n"] / L, color="#fff176", ls="--", lw=1, label=f"expected if random ({c['n'] / L:.1f})")
    ax_c.set_xlabel("Leaf index"); ax_c.set_ylabel("Times the worst leaf"); ax_c.set_title("How often each leaf was the worst", fontsize=10); _legend(ax_c, loc="upper right")
    ax_m.errorbar(c["mean"], range(L), xerr=2 * np.nan_to_num(c["sem"]), fmt="o", ms=3, color="#81c784", ecolor="#555", elinewidth=1)
    ax_m.axvline(0, color="#999", lw=1); ax_m.invert_yaxis(); ax_m.set_xlabel("Mean error ± 2 SE (mm), tilt removed"); ax_m.set_ylabel("Leaf index"); ax_m.set_title("Persistent offset per leaf", fontsize=10)
    fig.tight_layout()
    return _png(fig)


# ── entry point ───────────────────────────────────────────────────────────────────────────────────────────────────────

def build(db_path, machine: str, key: str, refs: dict, limits: dict) -> dict:
    """{'png': bytes | None, 'summary': str, 'headers': [...], 'rows': [[..., colour]]}. `refs` = {'mlc': mm, 'jaw': mm}; `limits` = {'walk': tol, 'mlc': (warn, fail), 'jaw': (warn, fail), 'pf': (warn, fail)}."""
    data = load(db_path, machine)
    wl, pf = data["wl"], data["pf"]
    out = {"png": None, "summary": "", "headers": [], "rows": []}
    if key == "walk":
        if not wl:
            out["summary"] = f"No Winston-Lutz records yet for {machine}."; return out
        out["png"], d = fig_walk(wl, limits["walk"])
        out["summary"] = "Radiation isocenter radius.  " + steps_text(d.get("steps")) + d["text"]
        out["headers"] = ["Date", "Max displacement (mm)", "Result"]
        tol = limits["walk"]
        for r in wl:
            v = r["walk_circle_r"]; lv = v <= tol
            out["rows"].append([f"{r['when']:%Y-%m-%d %H:%M}", f"{v:.3f}", "PASS" if lv else "FAIL", GRN if v <= 0.9 * tol else AMB if lv else RED])
    elif key in ("mlc", "jaw"):
        col = "avg_mlc_mm" if key == "mlc" else "avg_jaw_mm"
        rows = [r for r in wl if r.get(col) is not None]
        label = "MLC" if key == "mlc" else "Jaw"
        if not rows:
            out["summary"] = f"No {label} field-size records yet for {machine}."; return out
        warn, fail = limits[key]; ref = refs[key]
        out["png"], d = fig_size(wl, key, ref, warn, fail, label)
        y = np.array([r[col] - ref for r in rows])
        out["summary"] = f"{label} size minus reference ({ref:.2f} mm).  " + steps_text(d.get("steps")) + d["text"] + f"  Session-to-session scatter (SD) {float(np.std(y, ddof=1)) if len(y) > 1 else 0:.3f} mm."
        out["headers"] = ["Date", f"{label} (mm)", f"Δ {label} (mm)", "Result"]
        for r in rows:
            dv = r[col] - ref; a = abs(dv); lv = 0 if a <= warn else 1 if a <= fail else 2
            out["rows"].append([f"{r['when']:%Y-%m-%d %H:%M}", f"{r[col]:.3f}", f"{dv:+.3f}", ["PASS", "WARNING", "FAIL"][lv], [GRN, AMB, RED][lv]])
    elif key == "pf":
        if not pf:
            out["summary"] = f"No picket-fence records yet for {machine} (they are saved with the report, or by Import all new sessions)."; return out
        warn, fail = limits["pf"]
        out["png"], d = fig_pf(pf, warn, fail)
        c = leaf_consistency(pf)
        out["summary"] = "Largest leaf error.  " + steps_text(d.get("steps")) + d["text"] + "  " + leaf_text(c)
        out["headers"] = ["Date", "Max (mm)", "RMS (mm)", "Worst leaf", "Its error (mm)", "Result"]
        for r in pf:
            lv = 0 if r["max_dev_mm"] < warn else 1 if r["max_dev_mm"] < fail else 2
            out["rows"].append([f"{r['when']:%Y-%m-%d %H:%M}", f"{r['max_dev_mm']:.3f}", f"{r['rms_dev_mm']:.3f}", f"{r['worst_leaf']}", f"{r['worst_dev_mm']:+.3f}", ["PASS", "WARNING", "FAIL"][lv], [GRN, AMB, RED][lv]])
    elif key == "leaf":
        c = leaf_consistency(pf)
        out["summary"] = leaf_text(c)
        if not c.get("n"):
            return out
        out["png"] = fig_leaf(c)
        out["headers"] = ["Leaf", "y (mm)", "Times worst", "Mean error (mm)", "SE (mm)", "Persistent?"]
        for i in np.argsort(-np.array([c["counts"].get(i, 0) for i in range(c["L"])]))[:12]:
            i = int(i); pers = abs(c["z"][i]) >= 4 and abs(c["mean"][i]) >= 0.05
            out["rows"].append([f"{i}", f"{c['ys'][i]:+.1f}", f"{c['counts'].get(i, 0)}", f"{c['mean'][i]:+.3f}", f"{c['sem'][i]:.3f}", "YES" if pers else "no", RED if pers else GRN])
    return out
