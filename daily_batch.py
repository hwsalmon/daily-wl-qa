"""Import many Daily WL sessions at once (e.g. every session on the network EPID share since the start of the programme) without risking the application.

* One session at a time, each in its own try/except: a bad session is reported and the run goes on (nothing here raises for a data problem).
* Resumable: a saved session is registered (`epid_processed`), so a stopped or crashed run simply continues with what is left; nothing is analysed twice.
* Cancelable between images (`should_cancel`), with a `progress(done, total, text)` callback so the window stays alive.
* Light on memory: the big image arrays are dropped as soon as a session is analysed.
* Works with or without machine logs: the log folder is read ONCE per run, a missing / empty / unreadable folder or a log that does not match simply means "analysed without a log".
* A session is only SAVED when it is complete (four angles) and the void was found reliably at every angle; others are listed with the reason and left for review.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import traceback
from dataclasses import dataclass
from pathlib import Path

HEAVY_KEYS = ("pixel_array",)
PF_HEAVY_KEYS = ("_image_array", "_inv_array")


@dataclass
class Outcome:
    when: dt.datetime
    status: str            # saved | skipped | failed
    message: str = ""
    walk_mm: float | None = None
    passed: bool | None = None
    pf: bool = False
    logs: int = 0


def open_log_index(folders, cache_dir):
    """The log index of the first folder that has logs, or None. Never raises."""
    try:
        import wl_sources
    except Exception:
        return None
    for f in folders:
        if not f or not Path(f).is_dir():
            continue
        try:
            idx = wl_sources.LogIndex(f, cache_dir)
            if idx.records:
                return idx
        except Exception:
            continue
    return None


def analyse_session(t, sess, log_index=None, use_logs: bool = True) -> dict:
    """Analyse one session; returns {"ir", "wl", "pf"} (arrays dropped) or raises ValueError('reason') for a session that must not be saved."""
    import wl_sources
    ir = {a: t.analyze_image(t.read_image(p)) for a, p in sess.files.items()}
    for r in ir.values():
        for k in HEAVY_KEYS:
            r.pop(k, None)
    bad = [a for a, r in ir.items() if not (r.get("void_fit") or {}).get("ok", True)]
    if bad:
        raise ValueError(f"the void was not found reliably at G{', G'.join(str(a) for a in sorted(bad))} (no phantom, or a different test)")
    n_logs = 0
    if use_logs and log_index is not None:
        try:
            got = log_index.match({a: (r["when"], r["gantry"]) for a, r in ir.items() if r.get("when")})
            for a, rec in got.items():
                info = wl_sources.aperture_shift(rec)
                if info:
                    ir[a]["log"] = info; n_logs += 1
        except Exception:
            pass                                                    # a log problem means 'analysed without logs', never an error
    wl = t.compute_wl_results(ir, use_logs=use_logs)
    pf = None
    if sess.pf:
        try:
            ds = t.read_image(sess.pf)
            pf = t.analyze_picket_fence(ds)
            for k in PF_HEAVY_KEYS:
                pf.pop(k, None)
        except Exception:
            pf = None                                               # no usable picket-fence image: the Winston-Lutz result is still saved
    return {"ir": ir, "wl": wl, "pf": pf, "logs": n_logs}


def import_sessions(t, sessions, machine: str, physicist: str, *, log_folders=(), cache_dir: Path | None = None, use_logs: bool = True,
                    progress=None, should_cancel=None, dry_run: bool = False) -> list[Outcome]:
    """Analyse and save `sessions` (oldest first). `t` is the wl_qa_tool module. Returns one Outcome per session handled; a cancelled run returns what it did."""
    out: list[Outcome] = []
    ordered = sorted(sessions, key=lambda s: s.when)
    idx = open_log_index(log_folders, cache_dir or Path(".")) if use_logs else None
    total = len(ordered)
    for i, s in enumerate(ordered):
        if should_cancel and should_cancel():
            break
        if progress:
            progress(i, total, f"Session {i + 1} of {total}: {s.when:%Y-%m-%d %H:%M}")
        if not s.complete:
            out.append(Outcome(s.when, "skipped", f"images missing at G{', G'.join(str(a) for a in s.missing)}")); continue
        try:
            res = analyse_session(t, s, idx, use_logs)
        except ValueError as e:
            out.append(Outcome(s.when, "skipped", str(e))); continue
        except Exception as e:                                       # anything else in one session: report it and carry on
            out.append(Outcome(s.when, "failed", f"{type(e).__name__}: {e}")); continue
        wl = res["wl"]
        try:
            if not dry_run:
                t._save_to_db(wl, res["ir"], machine, physicist, when=s.when, pf_results=res["pf"])
        except Exception as e:
            out.append(Outcome(s.when, "failed", f"could not save: {type(e).__name__}: {e}")); continue
        out.append(Outcome(s.when, "saved", "", wl["max_2d_walk_mm"], bool(wl["pass_fail"]), res["pf"] is not None, res["logs"]))
    if progress:
        progress(total, total, "Done")
    return out


def summary_text(outcomes: list[Outcome]) -> str:
    saved = [o for o in outcomes if o.status == "saved"]
    skipped = [o for o in outcomes if o.status == "skipped"]
    failed = [o for o in outcomes if o.status == "failed"]
    lines = [f"{len(saved)} session(s) saved ({sum(o.pf for o in saved)} with a picket fence, {sum(1 for o in saved if o.logs)} with machine logs), "
             f"{len(skipped)} left for review, {len(failed)} failed."]
    fails = [o for o in saved if o.passed is False]
    if fails:
        lines.append("Winston-Lutz FAIL: " + ", ".join(f"{o.when:%Y-%m-%d} ({o.walk_mm:.2f} mm)" for o in fails))
    for o in skipped + failed:
        lines.append(f"  {o.when:%Y-%m-%d %H:%M}  {o.status}: {o.message}")
    return "\n".join(lines)
