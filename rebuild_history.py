"""Rebuild the Daily WL history database from the images themselves.

Every complete Daily WL session found under the given folders (DICOM RTIMAGE or native iView exports) is analysed with the current methods (log correction when the
machine logs are in the session's folder) and written to a NEW history database with the date and time the images were ACQUIRED. The old database is kept next to it as
`<name>.bak-before-rebuild-<date>` and the new one is built beside it and swapped in only when finished.

    python rebuild_history.py [--db wl_qa_history.db] [--physicist "Name"] [folder ...]      (default folders: "WL Test Data" and "Test Data")
    python rebuild_history.py --dry-run                                                      (analyse and report, write nothing)
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--db", default=str(HERE / "wl_qa_history.db"))
    ap.add_argument("--physicist", default="Howard W. Salmon, PhD, DABR (re-analysed from images)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    import wl_qa_tool as t
    import wl_sources as ws

    folders = [Path(f) for f in (a.folders or [HERE / "WL Test Data", HERE / "Test Data"])]
    cache = Path(tempfile.mkdtemp(prefix="wl_rebuild_"))
    sessions = {}
    for f in folders:
        if f.is_dir():
            for s in ws.scan_epid_folder(f, cache):
                if s.complete:
                    sessions[s.key] = s                          # the same session in two folders counts once
    rows, skipped = [], []
    for key, s in sorted(sessions.items(), key=lambda kv: kv[1].when):
        first = Path(next(iter(s.files.values())))
        machine = t.PATIENT_ID_MACHINE_MAP.get(s.patient) or t.PATIENT_ID_MACHINE_MAP.get(t._native_patient(first.parent if first.is_dir() else first.parent) or "", "")
        if not machine:
            skipped.append((key, "machine unknown")); continue
        try:
            ir = {ang: t.analyze_image(t.read_image(p)) for ang, p in s.files.items()}
            bad = [ang for ang, r in ir.items() if not (r.get("void_fit") or {}).get("ok", True)]
            if bad:
                skipped.append((key, f"void not found reliably at G{bad}")); continue
            logdir = first.parent if first.is_dir() else None          # native export: the logs sit beside the img_ folders
            if logdir is not None:
                idx = ws.LogIndex(logdir, cache)
                if idx.records:
                    for ang, rec in idx.match({ang: (r["when"], r["gantry"]) for ang, r in ir.items() if r.get("when")}).items():
                        info = ws.aperture_shift(rec)
                        if info:
                            ir[ang]["log"] = info
            wl = t.compute_wl_results(ir, use_logs=True)
            pf = None
            if s.pf:
                try:
                    pf = t.analyze_picket_fence(t.read_image(s.pf))
                    for k in ("_image_array", "_inv_array"):
                        pf.pop(k, None)
                except Exception:
                    pf = None
            for r in ir.values():
                r.pop("pixel_array", None)
        except Exception as e:                                           # one bad session must not stop the rebuild
            skipped.append((key, f"{type(e).__name__}: {e}")); continue
        rows.append((s, machine, ir, wl, pf))
        print(f"{s.when:%Y-%m-%d %H:%M}  {machine:24s} max displacement {wl['max_2d_walk_mm']:.2f} mm  ({'logs' if wl['logs_used'] else 'no logs'})", flush=True)
    print(f"\n{len(rows)} sessions analysed, {len(skipped)} skipped")
    for k, why in skipped:
        print("  skipped", k, "-", why)
    if a.dry_run or not rows:
        return 0
    db = Path(a.db)
    stamp = dt.date.today().strftime("%Y%m%d")
    if db.exists():
        shutil.copy2(db, db.with_name(f"{db.name}.bak-before-rebuild-{stamp}"))
    new = db.with_name(db.name + ".new")
    new.unlink(missing_ok=True)
    old_path, t.DB_PATH = t.DB_PATH, str(new)
    try:
        t._init_db()
        for s, machine, ir, wl, pf in rows:
            t._save_to_db(wl, ir, machine, a.physicist, when=s.when, pf_results=pf)
    finally:
        t.DB_PATH = old_path
    new.replace(db)
    n = sqlite3.connect(db).execute("select count(*) from wl_records").fetchone()[0]
    npf = sqlite3.connect(db).execute("select count(*) from pf_records").fetchone()[0]
    print(f"\nrebuilt {db} with {n} records and {npf} picket-fence records (backup: {db.name}.bak-before-rebuild-{stamp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
