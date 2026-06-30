# Winston-Lutz Daily QA Tool

Standalone Python GUI for daily Winston-Lutz (WL) mechanical isocenter QA on
Elekta Versa HD LINACs using the Standard Imaging MIMI Phantom (6.4 mm air void).

## Running the app

```bash
DISPLAY=:0 python3 wl_qa_tool.py
```

No server, no build step. Requires a display (Xwayland works on COSMIC/Wayland).

## Dependencies

```bash
pip install PySide6 pydicom opencv-python scipy reportlab matplotlib pillow
```

All imports are checked at startup; missing packages are reported with the exact
`pip install` command and the process exits.

## Machine configuration

Three Elekta Versa HD LINACs are configured in `MACHINES` (line ~115):

| Serial | Name in app |
|--------|-------------|
| 153991 | Elekta VersaHD 153991 |
| 156724 | Elekta VersaHD 156724 |
| 154613 | Elekta VersaHD 154613 |

Three physicists are configured in `PHYSICISTS` (line ~121):
- Howard W. Salmon, PhD, DABR
- Shawn Hollars, MS, DABR
- Logen Hall, MS, DABR

`PATIENT_ID_MACHINE_MAP` (line ~132) maps the Elekta iViewGT `PatientID` DICOM
tag to a machine name for automatic dropdown selection on directory load:

```python
PATIENT_ID_MACHINE_MAP = {
    "QA_Daily_V1_26": "Elekta VersaHD 153991",
    "QA_Daily_V2_26": "Elekta VersaHD 156724",
    "QA_Daily_MV_26": "Elekta VersaHD 154613",
}
```

`PatientID` is the only reliable machine identifier in iViewGT DICOM headers —
there is no serial number or station name tag.

To add machines or physicists, edit those three structures — no other code changes needed.

## DICOM input format

Elekta iViewGT RTIMAGEs **lack a File Meta Information header** (Transfer Syntax
UID is absent). All DICOM loads use `pydicom.dcmread(..., force=True)`.

Files are matched to cardinal gantry angles by reading the `GantryAngle` tag and
rounding to the nearest of G0/G90/G180/G270, within ±5°.

## Pixel convention (critical)

Elekta iViewGT stores pixels **inverted**: LOW value = HIGH dose.

- Background (outside field): ~65 000
- Irradiated field: ~24 000–28 000
- Air void (highest dose): local MINIMUM within the field (~24 200)

Detection therefore looks for the **darkest spot** inside the field, not the brightest.

## Void detection algorithm

1. Gaussian pre-filter (σ = void\_radius\_px × 0.30) — suppresses MV portal noise.
2. Locate the global minimum pixel within the central 50% of the field (10 mm radius).
3. Sub-pixel centroid using inverse-intensity-squared weighting in a local window.

SAD/SID magnification correction: `spacing_iso = spacing_det × (SAD / SID)`.
For the Elekta Versa HD: SAD = 1000 mm, SID = 1600 mm → scale factor = 0.625.

## Portal coordinate system (critical)

Elekta iViewGT stores portal images with **consistent patient orientation** at all
gantry angles (no image flip at opposing angles). The portal-plane axes map to
patient space as follows:

| Gantry | Portal X | Portal Y |
|--------|----------|----------|
| G0° / G180° | Patient **Lateral** (L/R) | Patient **SI** (S/I) |
| G90° / G270° | Patient **AP** (A/P) | Patient **SI** (S/I) |

Portal Y is always SI because the gantry rotation axis is the patient's SI axis.

## Walk circle metric (PASS/FAIL)

The residual CBCT setup error is decomposed into 3D patient coordinates using all
four gantry angles:

```
Lateral  = (ΔX_G0  + ΔX_G180) / 2
SI       = (ΔY_G0  + ΔY_G90 + ΔY_G180 + ΔY_G270) / 4
AP       = (ΔX_G90 + ΔX_G270) / 2
```

Subtracting the angle-appropriate component from each raw displacement yields
per-angle mechanical walk residuals. The **Minimum Enclosing Circle (MEC)** of
the four corrected walk vectors gives the walk circle metric.

**PASS: walk circle radius ≤ 1.0 mm** (`TOLERANCE_MM` constant).

## Trend database

Results are saved to `wl_qa_history.db` (SQLite, same directory as the script)
every time a PDF report is generated. The `View Trends` button plots walk circle
radius over time per machine.

The DB is created automatically on first run. Schema: `wl_records` table with
date, time, machine\_name, physicist\_name, walk\_circle\_r, pass\_fail,
baseline\_dx (lateral mm), baseline\_dy (SI mm), baseline\_dz (AP mm),
and raw ΔX/ΔY for each of the four gantry angles.
Older databases are migrated automatically by `_init_db()` (ALTER TABLE ADD COLUMN).

## PDF report

Generated via reportlab.  Always 3 pages.

**Page 1 — WL results**
- Title, metadata table: date/time from DICOM `StudyDate`/`StudyTime` (never today),
  machine, physicist, field size, void diameter, tolerance, 3D CBCT setup error
- PASS/FAIL colour banner with walk circle radius
- 2-column row: corrected displacement table | walk circle figure
- WL methodology notes

**Page 2 — Field size results**
- Field size heading, FS PASS/FAIL banner
- 2-column row: angle field-size table | leaf span table
- Field size methodology notes

**Page 3 — Portal images & signature**
- 5-panel diagnostic figure (4 portal images + displacement map)
- Caption
- Electronic signature block: machine, physicist, study date/time, result, 21 CFR Part 11 statement

**Study date handling (critical):**
- `generate_pdf_report()` receives `dicom_date` and `dicom_time` parameters.
- Date comes from DICOM `StudyDate` → `ContentDate` → folder name parse (MMDDYYYY / MMDDYY).
  `datetime.today()` is **never** used as a date source.
- PDF internal `CreationDate`/`ModDate` are set via a custom `_StudyTS` class that
  replaces `document._timeStamp` (reportlab 4.4.x internal).
- Filesystem `mtime`/`atime` are back-dated to study date via `os.utime()`.

## Diagnostic figure

Five panels generated by matplotlib Agg backend (no display required):
- Panels 1–4: portal image crop for each cardinal angle, windowed to field-interior
  pixels (2nd–90th percentile of pixels where normalised value < 0.15).
  Overlays: cyan field boundary, crosshair, coloured void marker, yellow
  displacement arrow, white 5 mm scale bar.
- Panel 5: 2-D displacement map — coloured dots for each angle, white MEC walk
  circle, blue MEC centre cross.

Figure is saved as a temp PNG and embedded in the GUI (Portal Images tab) and PDF.
The GUI renders it via `QPixmap.loadFromData()` (PySide6).

## Key constants (top of wl_qa_tool.py)

| Constant | Value | Meaning |
|----------|-------|---------|
| `TOLERANCE_MM` | 1.0 | PASS/FAIL threshold (mm) |
| `VOID_DIAMETER_MM` | 6.4 | MIMI air void diameter (mm) |
| `FIELD_SIZE_MM` | 40.0 | Nominal field size (mm) |
| `VOID_SEARCH_HALF_FIELD_FRACTION` | 0.50 | Search radius = 50% of field half-width |
| `GAUSSIAN_SIGMA_FRACTION` | 0.30 | Pre-filter σ as fraction of void radius |

## File layout

```
wl_qa_tool.py              — entire application (single file)
batch_generate_reports.py  — batch PDF generator: all sessions, all machines
install.py                 — one-step Linux/macOS installer
run_wl_qa.bat              — Windows double-click launcher
setup_windows.bat          — Windows first-time setup (embeddable Python)
requirements.txt           — pip dependency list
wl_qa_history.db           — SQLite trend database (auto-created, not in git)
wl_qa_config.json          — last-used paths (auto-created, not in git)
CLAUDE.md                  — this file
.gitignore                 — excludes __pycache__, *.pyc, *.pdf, *.db, WL Test Data/
Test Data/                 — sample DICOM directories (not committed)
WL Test Data/              — clinical DICOM sessions (not committed)
```
