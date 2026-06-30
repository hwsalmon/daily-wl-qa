#!/usr/bin/env python3
"""
Batch WL QA report generator.
Processes all sessions from 2026-06-19 to today for all three machines
and saves PDF reports to the same WL Test Data directory.
"""

import sys, os, glob, re, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Minimal Qt application required for matplotlib/PySide6 imports inside wl_qa_tool
os.environ.setdefault("DISPLAY", ":0")
from PySide6.QtWidgets import QApplication
_app = QApplication.instance() or QApplication(sys.argv)

import wl_qa_tool as wl

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WL Test Data")
OUTPUT_DIR = BASE_DIR
CUTOFF     = datetime.date(2026, 6, 19)

# Physicist rule: Shawn Hollars for Jun 20–26, Howard Salmon otherwise
def physicist_for_date(d: datetime.date) -> str:
    if datetime.date(2026, 6, 20) <= d <= datetime.date(2026, 6, 26):
        return "Shawn Hollars, MS, DABR"
    return "Howard W. Salmon, PhD, DABR"

# ── Find sessions ──────────────────────────────────────────────────────────────

def parse_folder_date(folder_name: str):
    # Try MMDDYYYY (8-digit)
    m = re.search(r"(\d{2})(\d{2})(\d{4})", folder_name)
    if m:
        try:
            return datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass
    # Try MMDDYY (6-digit)
    m = re.search(r"(\d{2})(\d{2})(\d{2})(?!\d)", folder_name)
    if m:
        try:
            return datetime.date(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass
    return None

# Collect unique directories: deduplicate by resolved path
seen_paths = set()
sessions = []  # (study_date, machine_name, directory, study_time)

for pattern in ("Daily WL MV *", "Daily WL V1 *", "Daily WL V2 *",
                "Daily QA WL MV *", "Daily QA WL V1 *", "Daily QA WL V2 *"):
    for d in sorted(glob.glob(os.path.join(BASE_DIR, pattern))):
        real = os.path.realpath(d)
        if real in seen_paths:
            continue
        seen_paths.add(real)

        # Need at least 4 DICOM files
        dcms = glob.glob(os.path.join(d, "**", "*.dcm"), recursive=True)
        dcms += [f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                 if os.path.isfile(f) and os.path.splitext(f)[1] == ""]
        if len(dcms) < 4:
            continue

        # Read one DICOM to get PatientID and study date
        import pydicom
        try:
            ds = pydicom.dcmread(dcms[0], force=True)
        except Exception:
            continue

        pid     = str(getattr(ds, "PatientID", "") or "")
        machine = wl.PATIENT_ID_MACHINE_MAP.get(pid)
        if not machine:
            continue

        study_date = None
        study_time = None
        for dtag, ttag in (("StudyDate", "StudyTime"), ("ContentDate", "ContentTime")):
            raw_d = getattr(ds, dtag, None)
            if raw_d and len(str(raw_d)) == 8:
                try:
                    study_date = datetime.datetime.strptime(str(raw_d), "%Y%m%d").date()
                    raw_t = getattr(ds, ttag, None)
                    if raw_t:
                        ts = str(raw_t).split(".")[0].zfill(6)
                        if len(ts) >= 6:
                            study_time = datetime.time(
                                int(ts[0:2]), int(ts[2:4]), int(ts[4:6])
                            )
                except (ValueError, TypeError):
                    pass
                if study_date:
                    break

        if not study_date:
            study_date = parse_folder_date(os.path.basename(d))

        if study_date and study_date >= CUTOFF:
            sessions.append((study_date, machine, d, study_time))

sessions.sort(key=lambda x: (x[0], x[1]))

print(f"Found {len(sessions)} sessions to process:\n")
for sd, machine, d, st in sessions:
    physicist = physicist_for_date(sd)
    print(f"  {sd}  {machine}  [{physicist.split(',')[0]}]  {os.path.basename(d)}")

print()

# ── Process each session ───────────────────────────────────────────────────────

import tempfile
ok, failed = 0, []

for study_date, machine, directory, study_time in sessions:
    physicist = physicist_for_date(study_date)
    date_str  = study_date.strftime("%Y%m%d")
    pdf_name  = f"WL_QA_{machine.replace(' ', '_')}_{date_str}.pdf"
    pdf_path  = os.path.join(OUTPUT_DIR, pdf_name)

    print(f"Processing  {study_date}  {machine}  →  {pdf_name} ...", end="  ", flush=True)

    try:
        # Load and analyse DICOM images
        dcm_images   = wl.load_dicom_images(directory)
        image_results = {a: wl.analyze_image(ds) for a, ds in dcm_images.items()}
        wl_results    = wl.compute_wl_results(image_results)

        # Generate figures to temp files
        diag_fig_path = wl.generate_diagnostic_figure(image_results, wl_results)
        walk_fig_path = wl.generate_walk_circle_figure(wl_results)

        # Save trend DB entry
        wl._save_to_db(wl_results, image_results, machine, physicist)

        # Generate PDF
        wl.generate_pdf_report(
            wl_results,
            image_results,
            pdf_path,
            diag_fig_path=diag_fig_path,
            walk_fig_path=walk_fig_path,
            machine_name=machine,
            physicist_name=physicist,
            dicom_date=study_date,
            dicom_time=study_time,
        )

        result = "PASS" if wl_results["pass_fail"] else "FAIL"
        walk_r = wl_results["max_2d_walk_mm"]
        print(f"{result}  walk={walk_r:.3f} mm")
        ok += 1

    except Exception as exc:
        print(f"ERROR: {exc}")
        failed.append((study_date, machine, str(exc)))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Completed: {ok}/{len(sessions)}  ({len(failed)} failed)")
if failed:
    print("Failed sessions:")
    for sd, m, err in failed:
        print(f"  {sd}  {m}  →  {err}")
