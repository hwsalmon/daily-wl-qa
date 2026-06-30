# Changelog

All notable changes to the Daily WL QA Tool are documented here.

---

## [Unreleased]

---

## 2026-06-30 — Picket Fence MLC leaf position QA

### Added
- **Picket Fence QA tab**: Analyses a single 1 cm MLC picket imaged on the EPID
  (SID 1600 mm, central 80 Agility leaves, 5 mm each, 20 cm field length).
  - Auto-detected when a PF DICOM is present in the same WL directory (identified
    by `RTImageLabel` or `SeriesDescription` containing "PF").
  - Manually loadable via **Load Picket Fence** button (separate folder supported).
  - Per-leaf centre deviation is shown in a portal image overlay + horizontal bar chart.
  - Stats card: N leaves, Max |Δ|, RMS, average measured width.
  - PASS/FAIL banner (green/red) appears alongside the WL and Field Size banners
    only on days when PF data is present.
- **PDF PF page**: When PF data is loaded, the report gains an additional page
  (between Field Size and Portal Images) showing the PASS/FAIL banner, summary
  stats table, diagnostic figure, and methodology note.  On non-PF days the page
  is absent — the report stays at 3 pages.
- **`PF_TOLERANCE_MM`** constant (1.5 mm) and `PF_LEAF_WIDTH_MM` (5 mm) at top of
  `wl_qa_tool.py`.
- **Algorithm**: Gaussian pre-filter → column-band row detection → per-leaf 50%
  penumbra edge finding → deviation = leaf centre − mean of all leaf centres
  (removes residual phantom offset).

---

## 2026-06-30 — Study-date PDF, machine auto-detection, 3-page report, batch processor

### Changed
- **Study date/time**: Reports always use `StudyDate`/`StudyTime` from the DICOM
  header.  `datetime.today()` is never used.  Fallback order:
  DICOM `StudyDate` → DICOM `ContentDate` → folder name (MMDDYYYY or MMDDYY pattern).
  If no date is recoverable, the date field is left blank rather than defaulting to
  today's date.
- **PDF filename**: Report filename now uses the DICOM study date
  (e.g. `WL_QA_Elekta_VersaHD_153991_20260619.pdf`) instead of the
  generation date.
- **PDF internal metadata**: `CreationDate` and `ModDate` fields inside the PDF
  are set to the study date/time (not the generation timestamp) via a custom
  `_StudyTS` object replacing `document._timeStamp` in reportlab 4.4.x.
  Filesystem `mtime`/`atime` are also back-dated via `os.utime()`.
- **Machine auto-detection**: The machine dropdown auto-selects the correct unit
  by reading the `PatientID` DICOM tag.  A new `PATIENT_ID_MACHINE_MAP` constant
  maps iViewGT PatientID strings to machine names:
  - `QA_Daily_V1_26` → Elekta VersaHD 153991
  - `QA_Daily_V2_26` → Elekta VersaHD 156724
  - `QA_Daily_MV_26` → Elekta VersaHD 154613
- **PDF report — 3-page layout**: Restructured from a variable-length document to
  a consistent 3-page format:
  - *Page 1*: metadata table, PASS/FAIL banner, corrected displacements + walk
    circle figure (side-by-side 2-column), WL methodology notes.
  - *Page 2*: field size heading, FS banner, field angle table + leaf span table
    (side-by-side 2-column), field size methodology notes.
  - *Page 3*: 5-panel portal diagnostic figure, caption, electronic signature block.
- **Electronic signature**: Timestamp in the signature block now uses study
  date/time, not the current time.

### Added
- **`batch_generate_reports.py`**: Standalone batch processor.  Scans `WL Test Data/`
  for all sessions matching the three configured machines (identified via `PatientID`),
  generates PDF reports and saves trend-DB entries for each.  Physicist assignment
  is date-driven (configurable at the top of the script).
  Usage: `python3 batch_generate_reports.py`

---

## 2026-06-09 — 3D patient-coordinate setup error correction

### Changed
- **`compute_wl_results`**: Setup error is now expressed as a proper 3D patient
  coordinate vector (Lateral / SI / AP mm) rather than mixed portal-image
  coordinates.  Elekta iViewGT stores portal images with consistent patient
  orientation at all gantry angles (no flip at G180/G270), so:
  - Lateral = (ΔX_G0 + ΔX_G180) / 2
  - SI = mean of all four ΔY values
  - AP = (ΔX_G90 + ΔX_G270) / 2
  Each angle's walk residual is corrected against its own physical axis before
  computing the MEC walk circle.  Walk circle radii are unchanged.
- **GUI**: Corrected-displacement card subtitle now shows live 3D CBCT setup
  error (Lat / SI / AP mm).  Column headers updated to "ΔLat/AP" and "ΔSI".
- **PDF report**: Metadata row shows 3D setup error with Lat/SI/AP breakdown;
  raw table labels which patient axis ΔX represents per angle; corrected table
  rows labelled "Lat walk" / "AP walk"; methodology notes rewritten.
- **SQLite DB**: Added `baseline_dz` column (AP setup error mm).  Existing
  databases are migrated automatically on first run.
- **CLAUDE.md / README**: Updated coordinate-system documentation throughout.

---

## 2026-06-07 — GUI migration from customtkinter to PySide6

### Changed
- **GUI framework**: Replaced customtkinter with PySide6.  PySide6 bundles its
  own Qt runtime and works with the embeddable Python zip used on Windows,
  whereas customtkinter requires tkinter/Tcl-Tk which is absent from embeddable
  Python.
- **Windows setup (`setup_windows.bat`)**: Reverted to the embeddable Python zip
  method (matching the rad-inventory app).  Downloads Python 3.12 embeddable,
  patches the `._pth` file to enable pip, installs PySide6 and all other
  dependencies — no admin rights, no system Python required.
- **Trend chart**: Rendered via matplotlib Agg backend to a PNG buffer displayed
  in a `QLabel`, avoiding any backend conflict with the main application.
- **requirements.txt**: `customtkinter` replaced with `PySide6>=6.6.0`.
- **README**: Updated package list, disk-space estimate (400 MB for PySide6),
  and Windows setup description.

---

## Earlier — Initial releases

- SQLite trend database with per-machine walk-circle time-series chart.
- Persistent config: last DICOM directory and report save location remembered
  across sessions (`wl_qa_config.json`).
- One-step Linux/macOS installer (`install.py`): installs dependencies, generates
  icon, and adds app to the desktop/application menu.
- Self-contained Windows setup (`setup_windows.bat`): no admin rights required.
- PDF report with electronic signature block (21 CFR Part 11 statement).
- Void detection algorithm: Gaussian pre-filter → global minimum → sub-pixel
  inverse-intensity-squared centroid.
- Magnification correction: `spacing_iso = spacing_det × (SAD/SID)`.
