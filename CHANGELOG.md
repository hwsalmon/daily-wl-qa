# Changelog

All notable changes to the Daily WL QA Tool are documented here.

---

## [Unreleased]

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
