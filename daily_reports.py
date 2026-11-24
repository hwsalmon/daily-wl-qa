"""Day, week and month reports from the stored history (no images needed), with the trend lines, and the electronic sign-off.

* `records_between()` : every session of a machine in a period, WL + field size + picket fence joined.
* `build_report()`    : the PDF (landscape A4): summary, the trend figures (with the period shaded), every session, for one day the per-angle residuals, the walk circle and the picket-fence
                        leaf errors, then the sign-off block. A draft says so; a signed report carries the physicist, the date and time, a comment and a SHA-256 digest of the data it covers.
* `add_signoff()` / `signoffs()` / `still_matches()` : the sign-offs are kept in the history database; a signed report can be re-checked later against the data (the digest changes if any
  record of the period was added, replaced or removed).
No Qt here.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import sqlite3
from pathlib import Path

import numpy as np

import daily_trends as tr

GANGLES = (0, 90, 180, 270)


# ── periods ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def period(kind: str, day: dt.date) -> tuple[dt.date, dt.date]:
    """(first, last) date of the day, the calendar week (Monday to Sunday) or the calendar month containing `day`."""
    if kind == "day":
        return day, day
    if kind == "week":
        a = day - dt.timedelta(days=day.weekday())
        return a, a + dt.timedelta(days=6)
    a = day.replace(day=1)
    b = (a.replace(year=a.year + 1, month=1) if a.month == 12 else a.replace(month=a.month + 1)) - dt.timedelta(days=1)
    return a, b


def label(kind: str, a: dt.date, b: dt.date) -> str:
    if kind == "day":
        return f"{a:%A %d %B %Y}"
    if kind == "week":
        return f"week {a:%G-W%V}  ({a:%d %b} to {b:%d %b %Y})"
    return f"{a:%B %Y}"


# ── data ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

def records_between(db_path, machine: str, a: dt.date, b: dt.date) -> list[dict]:
    """Sessions of `machine` from date a to b (inclusive), oldest first. Each WL record carries the picket fence of the same day ('pf') when there is one."""
    data = tr.load(db_path, machine)
    pf_by_day: dict = {}
    for r in data["pf"]:
        pf_by_day.setdefault(r["when"].date(), []).append(r)
    out = []
    for r in data["wl"]:
        if a <= r["when"].date() <= b:
            r = dict(r); r["pf"] = (pf_by_day.get(r["when"].date()) or [None])[-1]
            out.append(r)
    covered = {r["when"].date() for r in out}
    for d, rows in pf_by_day.items():                       # a picket fence with no Winston-Lutz record that day
        if a <= d <= b and d not in covered:
            out.append({"when": rows[-1]["when"], "walk_circle_r": None, "pass_fail": None, "pf": rows[-1], "date": rows[-1]["date"], "time": rows[-1]["time"], "machine_name": machine})
    return sorted(out, key=lambda r: r["when"])


def days_with_data(db_path, machine: str) -> list[dt.date]:
    data = tr.load(db_path, machine)
    return sorted({r["when"].date() for r in data["wl"]} | {r["when"].date() for r in data["pf"]})


def digest(records: list[dict]) -> str:
    """SHA-256 of the numbers a report covers (so a later change to any record of the period is detectable)."""
    keys = ("date", "time", "walk_circle_r", "pass_fail", "baseline_dx", "baseline_dy", "baseline_dz", "avg_mlc_mm", "avg_jaw_mm") + tuple(f"raw_d{c}_G{a}" for a in GANGLES for c in "xy")
    rows = []
    for r in records:
        pf = r.get("pf")
        rows.append([r.get(k) for k in keys] + ([pf["max_dev_mm"], pf["rms_dev_mm"], pf["worst_leaf"], pf.get("leaf_dev_json")] if pf else []))
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


# ── sign-off ledger ────────────────────────────────────────────────────────────────────────────────────────────────────

def ensure_tables(conn) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS report_signoffs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, machine_name TEXT NOT NULL, kind TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL,
        physicist TEXT NOT NULL, signed_at TEXT NOT NULL, comment TEXT, n_sessions INTEGER, digest TEXT NOT NULL, pdf_name TEXT)""")


def add_signoff(db_path, machine, kind, a, b, physicist, comment, records, pdf_name="") -> dict:
    c = sqlite3.connect(str(db_path))
    ensure_tables(c)
    now = dt.datetime.now().isoformat(timespec="seconds")
    d = digest(records)
    cur = c.execute("INSERT INTO report_signoffs (machine_name, kind, start_date, end_date, physicist, signed_at, comment, n_sessions, digest, pdf_name) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (machine, kind, a.isoformat(), b.isoformat(), physicist, now, comment or "", len(records), d, pdf_name))
    c.commit(); sid = cur.lastrowid; c.close()
    return {"id": sid, "machine": machine, "kind": kind, "start": a, "end": b, "physicist": physicist, "signed_at": now, "comment": comment or "", "n": len(records), "digest": d}


def signoffs(db_path, machine: str | None = None) -> list[dict]:
    try:
        c = sqlite3.connect(str(db_path)); c.row_factory = sqlite3.Row
        ensure_tables(c)
        q = "SELECT * FROM report_signoffs" + (" WHERE machine_name=?" if machine else "") + " ORDER BY id DESC"
        rows = [dict(r) for r in c.execute(q, (machine,) if machine else ())]
        c.close()
        return rows
    except sqlite3.Error:
        return []


def still_matches(db_path, so: dict) -> bool:
    a, b = dt.date.fromisoformat(so["start_date"]), dt.date.fromisoformat(so["end_date"])
    return digest(records_between(db_path, so["machine_name"], a, b)) == so["digest"]


# ── PDF ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _fmt(v, nd=2, sign=False):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")


def build_report(db_path, machine: str, kind: str, a: dt.date, b: dt.date, *, t, refs: dict, limits: dict, version: str, signoff: dict | None = None, out_path=None) -> bytes:
    """The report as PDF bytes (and written to `out_path` when given). `t` is the wl_qa_tool module (for the per-angle residual calculation)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    recs = records_between(db_path, machine, a, b)
    wl = [r for r in recs if r.get("walk_circle_r") is not None]
    pfs = [r["pf"] for r in recs if r.get("pf")]
    tol = limits["walk"]; (mw, mf), (jw, jf), (pw, pt) = limits["mlc"], limits["jaw"], limits["pf"]
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceAfter=2)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#0d47a1"))
    BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=11.5)
    SMALL = ParagraphStyle("S", parent=BODY, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#444444"))
    CELL = ParagraphStyle("C", parent=BODY, fontSize=8, leading=9.5)
    GRN, AMB, RED = colors.HexColor("#c8e6c9"), colors.HexColor("#fff3b0"), colors.HexColor("#ffcdd2")

    def table(rows, widths, extra=()):
        t_ = Table(rows, colWidths=widths, repeatRows=1)
        t_.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d47a1")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bdbdbd")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")])] + list(extra)))
        return t_

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm, topMargin=12 * mm, bottomMargin=14 * mm,
                            title=f"Daily QA - {label(kind, a, b)} - {machine}", author="Daily QA - WL - DLG - Picket Fence")
    el = []
    kname = {"day": "Daily", "week": "Weekly", "month": "Monthly"}[kind]
    el += [Paragraph(f"Daily QA - WL - DLG - Picket Fence: {kname} report", H1),
           Paragraph(f"<b>{machine}</b> &nbsp;|&nbsp; {label(kind, a, b)} &nbsp;|&nbsp; {len(wl)} Winston-Lutz session(s), {len(pfs)} picket fence(s)", BODY),
           Paragraph(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} &nbsp;|&nbsp; software v{version} &nbsp;|&nbsp; "
                     + ("<b>SIGNED</b>" if signoff else "<font color=red><b>DRAFT - not signed</b></font>")
                     + f" &nbsp;|&nbsp; limits: isocenter displacement &le; {tol:.1f} mm (SRS); MLC size warn/fail {mw:.1f}/{mf:.1f} mm; jaw size {jw:.1f}/{jf:.1f} mm; picket fence warn/fail {pw:.1f}/{pt:.1f} mm", SMALL),
           Spacer(1, 4)]
    if not recs:
        el.append(Paragraph("No results were recorded in this period.", BODY))

    # 1. summary
    if recs:
        el.append(Paragraph("1. Summary", H2))
        walk = np.array([r["walk_circle_r"] for r in wl], float)
        mlc = np.array([r["avg_mlc_mm"] - refs["mlc"] for r in wl if r.get("avg_mlc_mm") is not None], float)
        jaw = np.array([r["avg_jaw_mm"] - refs["jaw"] for r in wl if r.get("avg_jaw_mm") is not None], float)
        rows = [["Measure", "Sessions", "Mean", "Range", "Worst", "Pass", "Warn", "Fail"]]
        style = []

        def row(name, v, lim_fn, unit="mm", signed=False):
            if not len(v):
                return
            lv = [lim_fn(x) for x in v]
            worst = v[int(np.argmax(np.abs(v)))]
            rows.append([name, len(v), f"{np.mean(v):{'+' if signed else ''}.3f}", f"{v.min():{'+' if signed else ''}.2f} to {v.max():{'+' if signed else ''}.2f}", f"{worst:{'+' if signed else ''}.3f}",
                         lv.count(0), lv.count(1), lv.count(2)])
            i = len(rows) - 1
            if lv.count(2):
                style.append(("BACKGROUND", (7, i), (7, i), RED))
            if lv.count(1):
                style.append(("BACKGROUND", (6, i), (6, i), AMB))
        row("Isocenter: max displacement (mm)", walk, lambda x: 0 if x <= 0.9 * tol else 1 if x <= tol else 2)
        row("MLC size minus reference (mm)", mlc, lambda x: 0 if abs(x) <= mw else 1 if abs(x) <= mf else 2, signed=True)
        row("Jaw size minus reference (mm)", jaw, lambda x: 0 if abs(x) <= jw else 1 if abs(x) <= jf else 2, signed=True)
        if pfs:
            row("Picket fence: max leaf error (mm)", np.array([p["max_dev_mm"] for p in pfs], float), lambda x: 0 if x < pw else 1 if x < pt else 2)
            row("Picket fence: RMS leaf error (mm)", np.array([p["rms_dev_mm"] for p in pfs], float), lambda x: 0)
        el.append(table(rows, [70 * mm, 22 * mm, 24 * mm, 40 * mm, 24 * mm, 18 * mm, 18 * mm, 18 * mm], style))

    # 2. trends (the whole history up to the end of the period, the period shaded)
    if recs:
        el.append(Paragraph("2. Trend lines", H2))
        el.append(Paragraph("The history up to the end of this period (shaded: the period reported). Dashed lines are the limits; the straight line is the fitted drift (flagged only when statistically firm "
                            "and larger than the stated size); a dotted vertical line marks a step change.", SMALL))
        notes = []
        for key, title in (("walk", "Radiation isocenter"), ("jaw", "Jaw size"), ("mlc", "MLC size"), ("pf", "Picket fence")):
            try:
                r = tr.build(db_path, machine, key, refs, limits, until=b, shade=(a, b), compact=True)
            except Exception as e:
                r = {"png": None, "summary": f"not drawn ({type(e).__name__})"}
            if r["png"]:
                from PIL import Image as PILImage
                w_, h_ = PILImage.open(io.BytesIO(r["png"])).size
                width = min(250 * mm, 68 * mm * w_ / h_)
                el.append(KeepTogether([Paragraph(f"<b>{title}</b>: {r['summary']}", CELL), Image(io.BytesIO(r["png"]), width=width, height=width * h_ / w_)]))
                notes.append(title)
            elif r["summary"] and "No " not in r["summary"][:3]:
                el.append(Paragraph(f"<b>{title}</b>: {r['summary']}", CELL))
        if not notes:
            el.append(Paragraph("No trend could be drawn.", BODY))

    # 3. sessions
    if recs:
        el.append(PageBreak() if kind != "day" else Spacer(1, 6))
        el.append(Paragraph("3. Sessions", H2))
        rows = [["Date / time", "Isocenter (mm)", "WL", "Setup lat / SI / AP (mm)", "MLC size (mm)", "Δ MLC", "Jaw size (mm)", "Δ Jaw", "PF max / RMS (mm)", "Worst leaf", "Machine log"]]
        style = []
        for i, r in enumerate(recs, start=1):
            w_ = r.get("walk_circle_r"); pf = r.get("pf")
            dm = None if r.get("avg_mlc_mm") is None else r["avg_mlc_mm"] - refs["mlc"]
            dj = None if r.get("avg_jaw_mm") is None else r["avg_jaw_mm"] - refs["jaw"]
            rows.append([f"{r['when']:%Y-%m-%d %H:%M}", _fmt(w_, 3), "-" if w_ is None else "PASS" if w_ <= tol else "FAIL",
                         "-" if r.get("baseline_dx") is None else f"{r['baseline_dx']:+.2f} / {r['baseline_dy']:+.2f} / {r['baseline_dz']:+.2f}",
                         _fmt(r.get("avg_mlc_mm"), 2), _fmt(dm, 2, True), _fmt(r.get("avg_jaw_mm"), 2), _fmt(dj, 2, True),
                         "-" if not pf else f"{pf['max_dev_mm']:.2f} / {pf['rms_dev_mm']:.2f}", "-" if not pf else f"{pf['worst_leaf']} ({pf['worst_dev_mm']:+.2f})",
                         "-" if not r.get("logs_used") else f"{r['logs_used']} of 4"])
            if w_ is not None:
                style.append(("BACKGROUND", (2, i), (2, i), GRN if w_ <= tol else RED))
            if dm is not None:
                style.append(("BACKGROUND", (5, i), (5, i), GRN if abs(dm) <= mw else AMB if abs(dm) <= mf else RED))
            if dj is not None:
                style.append(("BACKGROUND", (7, i), (7, i), GRN if abs(dj) <= jw else AMB if abs(dj) <= jf else RED))
            if pf:
                style.append(("BACKGROUND", (8, i), (8, i), GRN if pf["max_dev_mm"] < pw else AMB if pf["max_dev_mm"] < pt else RED))
        el.append(table(rows, [32 * mm, 22 * mm, 14 * mm, 40 * mm, 22 * mm, 16 * mm, 22 * mm, 16 * mm, 30 * mm, 24 * mm, 20 * mm], style))

    # 4. one day: per-angle detail, walk circle, picket fence leaves
    if kind == "day" and wl:
        r = wl[-1]
        ir = {ang: {"dx_mm": r.get(f"raw_dx_G{ang}"), "dy_mm": r.get(f"raw_dy_G{ang}")} for ang in GANGLES}
        if all(v["dx_mm"] is not None and v["dy_mm"] is not None for v in ir.values()):
            res = t.compute_wl_results(ir, use_logs=False)
            el.append(Paragraph("4. Winston-Lutz detail", H2))
            rows = [["Gantry", "Raw lateral (mm)", "Raw SI (mm)", "After the static offset: lateral", "SI", "Displacement (mm)"]]
            for ang in GANGLES:
                p = res["per_angle"][ang]
                rows.append([f"G{ang}", f"{p['raw_dx']:+.3f}", f"{p['raw_dy']:+.3f}", f"{p['rel_dx']:+.3f}", f"{p['rel_dy']:+.3f}", f"{np.hypot(p['rel_dx'], p['rel_dy']):.3f}"])
            el.append(table(rows, [24 * mm, 36 * mm, 30 * mm, 56 * mm, 24 * mm, 36 * mm]))
            el.append(Paragraph(f"Static phantom set-up offset removed: lateral {res['setup_x']:+.3f}, SI {res['setup_y']:+.3f}, AP {res['setup_z']:+.3f} mm. Largest displacement from the isocenter {res['max_2d_walk_mm']:.3f} mm "
                                f"({'PASS' if res['pass_fail'] else 'FAIL'}, limit {tol:.1f} mm).", SMALL))
        pf = r.get("pf") or (pfs[-1] if pfs else None)
        if pf is not None and len(pf["devs"]):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 2.8))
            ys = pf["ys"] if len(pf["ys"]) == len(pf["devs"]) else np.arange(len(pf["devs"]))
            ax.bar(ys, pf["devs"], width=3.5, color=["#ef5350" if abs(v) >= pt else "#ffa726" if abs(v) >= pw else "#66bb6a" for v in pf["devs"]])
            ax.axhline(pt, color="#c62828", ls="--", lw=1); ax.axhline(-pt, color="#c62828", ls="--", lw=1); ax.set_xlabel("leaf position along y (mm)"); ax.set_ylabel("error (mm)")
            ax.set_title("Picket fence: error of every leaf", fontsize=9); ax.grid(alpha=0.3); fig.tight_layout()
            bb = io.BytesIO(); fig.savefig(bb, format="PNG", dpi=130); plt.close(fig)
            el += [Paragraph("5. Picket fence", H2), Image(io.BytesIO(bb.getvalue()), width=200 * mm, height=200 * mm * 2.8 / 9)]

    # sign-off
    el.append(Spacer(1, 8))
    block = [Paragraph("Physicist sign-off", H2)]
    if signoff:
        block += [Paragraph(f"<b>Electronically signed by {signoff['physicist']}</b> on {signoff['signed_at'].replace('T', ' ')} (sign-off no. {signoff['id']}). "
                            f"The physicist reviewed the {signoff['n']} session(s) of {label(kind, a, b)} on {machine} reported above.", BODY)]
        if signoff.get("comment"):
            block.append(Paragraph(f"Comment: {signoff['comment']}", BODY))
        block.append(Paragraph(f"Digest of the data covered (SHA-256): <font face='Courier' size=7>{signoff['digest']}</font>. The digest changes if any record of this period is later added, replaced or removed, "
                               "so the report can be checked against the database (History &gt; Reports).", SMALL))
    else:
        block.append(Paragraph("<font color=red><b>DRAFT - awaiting physicist review and electronic signature.</b></font>", BODY))
    el.append(KeepTogether(block))
    doc.build(el)
    pdf = buf.getvalue()
    if out_path:
        Path(out_path).write_bytes(pdf)
    return pdf
