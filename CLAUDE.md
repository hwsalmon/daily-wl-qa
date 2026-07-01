# Daily QA - WL - DLG - Picket Fence

Standalone Python GUI for daily Winston-Lutz (WL) mechanical isocenter QA on
Elekta Versa HD LINACs using the Standard Imaging MIMI Phantom (6.4 mm air void).
Also includes physical-jaw/MLC field-size (DLG) QA and Picket Fence MLC leaf
position QA on days it is performed.

`APP_NAME` (top of `wl_qa_tool.py`) holds the display name used in the window
title, header, dialogs, and PDF report/footer. "Winston-Lutz"/"WL" is kept in
methodology text and internal identifiers since it names the actual clinical
test, distinct from the app's branding.

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

## Machine and physicist configuration

`MACHINES` and `PHYSICISTS` (top of `wl_qa_tool.py`) are the **default** lists
used only on first run:

| Serial | Name in app |
|--------|-------------|
| 153991 | Elekta VersaHD 153991 |
| 156724 | Elekta VersaHD 156724 |
| 154613 | Elekta VersaHD 154613 |

- Howard W. Salmon, PhD, DABR
- Shawn Hollars, MS, DABR
- Logen Hall, MS, DABR

At runtime the app reads `self._machines` / `self._physicists`, seeded from
`config["machines"]` / `config["physicists"]` in `wl_qa_config.json` if
present, else falling back to the `MACHINES`/`PHYSICISTS` constants. **Adding
or removing a machine or physicist is done through the in-app Settings
dialog** (see below) — editing the module constants is only needed to change
the factory defaults for a fresh install.

`PATIENT_ID_MACHINE_MAP` (module constant, not settings-editable) maps the
Elekta iViewGT `PatientID` DICOM tag to a machine name for automatic dropdown
selection on directory load:

```python
PATIENT_ID_MACHINE_MAP = {
    "QA_Daily_V1_26": "Elekta VersaHD 153991",
    "QA_Daily_V2_26": "Elekta VersaHD 156724",
    "QA_Daily_MV_26": "Elekta VersaHD 154613",
}
```

`PatientID` is the only reliable machine identifier in iViewGT DICOM headers —
there is no serial number or station name tag. Adding a machine here still
requires a code change (and the machine name must also exist in the
Settings-managed machine list to appear in the dropdown).

## Settings dialog

Opened via the **Settings** button next to the Machine/Physicist dropdowns.
Three tabs, all implemented in `WLApp._show_settings()` and its
`_build_settings_*` helpers:

- **Machines** / **Physicists** — add/remove entries in a `QListWidget`.
  Changes persist immediately to `config["machines"]` / `config["physicists"]`
  in `wl_qa_config.json` via `_save_machines_physicists()`, and the main
  window's dropdowns refresh live while preserving the current selection.
  At least one entry must remain in each list.
- **Field Size Reference** — manual numeric MLC/Jaw reference entry per
  machine (`QDoubleSpinBox`, 3 decimal places), writing into
  `config["field_refs"][machine]`. This supplements — does not replace — the
  **Set Current as Reference** button on the Field Size QA tab, which derives
  the reference from a live measurement instead. Manual entry is for cases
  where the exact baseline is already known (e.g. after a physical
  calibration) and rounds to 3 decimals, same as the rest of the app's
  `.3f` display formatting.

## DICOM input format

Elekta iViewGT RTIMAGEs **lack a File Meta Information header** (Transfer Syntax
UID is absent). All DICOM loads use `pydicom.dcmread(..., force=True)`.

WL files are matched to cardinal gantry angles by reading the `GantryAngle` tag and
rounding to the nearest of G0/G90/G180/G270, within ±5°.

**PF DICOM identification (critical):** `identify_pf_dicom(ds)` checks whether
`RTImageLabel` or `SeriesDescription` contains "PF" or "PICKET" (case-insensitive).
`load_dicom_images()` calls this on every file and skips PF images before gantry-angle
matching — so the PF DICOM can safely sit in the same folder as the four WL images
without corrupting the G0 slot or the field-size analysis.

## Pixel convention (critical)

Elekta iViewGT stores pixels **inverted**: LOW value = HIGH dose.

- Background (outside field): ~65 000
- Irradiated field: ~24 000–28 000
- Air void (highest dose): local MINIMUM within the field (~24 200)

WL detection therefore looks for the **darkest spot** inside the field, not the brightest.
PF figure display uses the normalised-inverted array (high dose = bright pixel) so that
the picket appears as a bright stripe on a dark background.

## Void detection algorithm (WL)

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

## Field Size QA thresholds

MLC (leaf bank) and physical jaw deviations use **independent** action
levels — the jaw mechanism is less precise than the MLC leaves, so it gets a
wider tolerance:

| Axis | Warn | Fail | Constants |
|------|------|------|-----------|
| MLC (leaf banks) | 0.4 mm | 0.6 mm | `FIELD_SIZE_WARN_MM` / `FIELD_SIZE_FAIL_MM` |
| Physical jaw | 0.6 mm | 0.8 mm | `FIELD_SIZE_JAW_WARN_MM` / `FIELD_SIZE_JAW_FAIL_MM` |

`_fs_level(dev, kind)` (GUI) and `_fs_lvl(dev, kind)` / `_fcc(v, ref, kind)`
(PDF) take a `kind` of `"mlc"` or `"jaw"` to select the correct threshold
pair. Individual leaf-span deviations always use the MLC thresholds (leaves
are MLC hardware). The reference values themselves come from
`_get_field_refs(machine)` — either the **Set Current as Reference** button
(Field Size QA tab) or manual entry in the Settings dialog.

## Picket Fence QA

Performed on days the test is acquired (typically weekly or as scheduled).
Phantom: 1 cm MLC gap at G0°, EPID at SID = 1600 mm, central 80 Agility leaves
(40 per bank, 5 mm wide each at isocenter), 200 mm total picket length.

### PF detection & loading

- **Auto-detected**: on `Load DICOM Directory`, every file in the folder is tested
  with `identify_pf_dicom()`. The first PF file found is analysed automatically.
- **Manual load**: `Load Picket Fence` button opens a separate directory picker
  (for clinics that store PF images in a different folder).

### PF analysis algorithm

1. Gaussian pre-filter (σ = 1.5 px) on the normalised-inverted image.
2. Column-profile detection (mean along rows) to find the picket column band.
3. Row-profile detection using **max within the column band** (not global mean —
   avoids dilution by the narrow picket across 1024 background columns).
4. Leaf count capped at `PF_LEAVES_TOTAL // 2` (40): the 12% detection threshold
   sits in the penumbra, making the measured field slightly taller than 200 mm and
   causing `round()` to produce 41 without the cap.
5. Per-leaf 50%-penumbra edge finding on both sides of the picket → leaf centre.
6. Deviation = leaf centre − mean of all leaf centres (removes phantom offset).

### PF thresholds

| Level | Constant | Value | Colour |
|-------|----------|-------|--------|
| Pass | — | < `PF_WARN_MM` | Green |
| Warning | `PF_WARN_MM` | 0.4 mm | Amber |
| Fail | `PF_TOLERANCE_MM` | 0.5 mm | Red |

### PF figure display

Two-panel matplotlib figure (13 × 9 in):
- **Left** — portal image using the normalised-inverted array windowed from the
  central 90% of field rows (5th–95th percentile of that core). Background clips
  to black; end-of-picket penumbra and hot-spot artefacts are excluded from the
  window so the main picket body displays uniformly.
- **Right** — horizontal bar chart of per-leaf deviations; orange ±0.4 mm warning
  lines and red ±0.5 mm fail lines; X-axis zoomed to
  `max(2 × PF_TOLERANCE_MM, 1.5 × max_dev, 0.3 mm)`.

## Trend database

Results are saved to `wl_qa_history.db` (SQLite, same directory as the script)
every time a PDF report is generated. The `View Trends` button opens a
**machine picker → test picker → chart** flow (`_show_trends()` and its
`_build_trend_*_page()` helpers, using a `QStackedWidget`):

1. Pick a machine (only machines with existing DB records are listed).
2. Pick a test: Walk Circle Radius (WL), Field Size — MLC, or Field Size —
   Jaw. MLC/Jaw trends plot deviation from that machine's current reference
   (`_get_field_refs()`), colour-coded pass/warn/fail using the same
   thresholds as the GUI/PDF (see Field Size QA thresholds above).
3. Chart + table for that machine/test, with "Change Test"/"Change Machine"
   back-navigation. Stale pages are removed from the stack as new ones
   replace them (`stack.removeWidget(old); old.deleteLater()`) so repeated
   navigation doesn't leak widgets or leave back-buttons pointing at stale
   pages.

PF results are not yet stored in the trend DB (still a pending idea below),
so Picket Fence is not one of the selectable tests.

The DB is created automatically on first run. Schema: `wl_records` table with
date, time, machine\_name, physicist\_name, walk\_circle\_r, pass\_fail,
baseline\_dx (lateral mm), baseline\_dy (SI mm), baseline\_dz (AP mm),
and raw ΔX/ΔY for each of the four gantry angles.
Older databases are migrated automatically by `_init_db()` (ALTER TABLE ADD COLUMN).

## PDF report

3 pages on WL-only days; 4 pages on days when PF data is present.

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

**Page 3 (PF days only) — Picket Fence results**
- PF PASS/FAIL colour banner with Max |Δ| and tolerance
- 4-column stats table (leaves, max deviation, RMS, avg width)
- PF figure (7.10 × 4.80 in)
- PF methodology note (algorithm, thresholds)

**Page 3 or 4 — Portal images & signature**
- 5-panel diagnostic figure (4 portal images + displacement map)
- Caption
- Electronic signature block: machine, physicist, study date/time, WL result,
  field result, **PF result row** (when PF data is present), 21 CFR Part 11 statement

**Study date handling (critical):**
- `generate_pdf_report()` receives `dicom_date` and `dicom_time` parameters.
- Date comes from DICOM `StudyDate` → `ContentDate` → folder name parse (MMDDYYYY / MMDDYY).
  `datetime.today()` is **never** used as a date source.
- PDF internal `CreationDate`/`ModDate` are set via a custom `_StudyTS` class that
  replaces `document._timeStamp` (reportlab 4.4.x internal).
- Filesystem `mtime`/`atime` are back-dated to study date via `os.utime()`.

## Diagnostic figure (WL)

Five panels generated by matplotlib Agg backend (no display required):
- Panels 1–4: portal image crop for each cardinal angle. **Windowing is
  computed from a small ROI centred on the detected void** (± `5 ×` the
  void radius, 1st–97th percentile of that ROI only) rather than the whole
  crop — the void is a subtle local minimum riding on top of the much larger
  field→background intensity range, so windowing off the whole crop buried it
  in near-uniform dark gray. Void-local windowing saturates the field
  edge/background to black/white but renders the void as a clearly visible
  dark circle. Overlays: cyan field boundary, crosshair, coloured void
  marker, yellow displacement arrow, white 5 mm scale bar. (The magenta
  phantom-ring sanity circle that used to appear here was removed along with
  the ring-detection feature — see below.)
- Panel 5: 2-D displacement map — coloured dots for each angle, white MEC walk
  circle, blue MEC centre cross.

### Phantom ring detection — removed

`measure_phantom_ring()` (MIMI internal ring radius sanity check, ~22 mm) and
its GUI status label / diagnostic-figure overlay were removed. The check was
a secondary sanity indicator, not a PASS/FAIL criterion, and cluttered the
portal-image panels without adding QA value beyond the void/field detection
already performed. If ring-based phantom-tilt detection is wanted again,
the previous implementation (sector-gradient search between
`RING_SEARCH_MIN_MM`/`RING_SEARCH_MAX_MM`) is recoverable from git history
prior to the "Remove phantom ring detection" commit.

Figure is saved as a temp PNG and embedded in the GUI (Portal Images tab) and PDF.
The GUI renders it via `QPixmap.loadFromData()` (PySide6).

## Key constants (top of wl_qa_tool.py)

| Constant | Value | Meaning |
|----------|-------|---------|
| `APP_NAME` | "Daily QA - WL - DLG - Picket Fence" | App branding — window title, header, dialogs, PDF |
| `TOLERANCE_MM` | 1.0 | WL PASS/FAIL threshold (mm) |
| `VOID_DIAMETER_MM` | 6.4 | MIMI air void diameter (mm) |
| `FIELD_SIZE_MM` | 40.0 | Nominal WL field size (mm) |
| `VOID_SEARCH_HALF_FIELD_FRACTION` | 0.50 | Search radius = 50% of field half-width |
| `GAUSSIAN_SIGMA_FRACTION` | 0.30 | WL pre-filter σ as fraction of void radius |
| `FIELD_SIZE_REF_MM` | 40.0 | Nominal field-size reference before a machine is calibrated |
| `FIELD_SIZE_WARN_MM` / `FIELD_SIZE_FAIL_MM` | 0.4 / 0.6 | MLC field-size warn/fail (mm) |
| `FIELD_SIZE_JAW_WARN_MM` / `FIELD_SIZE_JAW_FAIL_MM` | 0.6 / 0.8 | Physical jaw field-size warn/fail (mm) |
| `PF_TOLERANCE_MM` | 0.5 | PF FAIL threshold per leaf (mm) |
| `PF_WARN_MM` | 0.4 | PF WARNING threshold per leaf (mm) |
| `PF_LEAF_WIDTH_MM` | 5.0 | Agility inner leaf width at isocenter (mm) |
| `PF_LEAVES_TOTAL` | 80 | Total leaves tested (40 per bank) |

## File layout

```
wl_qa_tool.py              — entire application (single file)
batch_generate_reports.py  — batch PDF generator: all sessions, all machines
install.py                 — one-step Linux/macOS installer
run_wl_qa.bat              — Windows double-click launcher
setup_windows.bat          — Windows first-time setup (embeddable Python)
requirements.txt           — pip dependency list
wl_qa_history.db           — SQLite trend database (auto-created, not in git)
wl_qa_config.json          — last-used paths, machines/physicists lists, and
                             per-machine field-size references (auto-created,
                             not in git)
CLAUDE.md                  — this file
.gitignore                 — excludes __pycache__, *.pyc, *.pdf, *.db,
                             WL Test Data/, PicketFence/
Test Data/                 — sample DICOM directories (not committed)
WL Test Data/              — clinical DICOM sessions (not committed)
PicketFence/               — PF sample DICOM (not committed — clinical data)
```

## Current application state (as of 2026-07-01)

### Features complete and tested
- WL void detection, walk circle, 3D CBCT setup error decomposition
- Field size QA (MLC/jaw deviation, leaf span) with independent MLC vs.
  physical-jaw warn/fail tolerances
- Machine auto-detection from DICOM `PatientID`
- Study-date PDF (DICOM StudyDate → filename, PDF metadata, filesystem mtime)
- 3-page PDF (WL / Field Size / Portal images + signature)
- SQLite trend database with a machine-picker → test-picker → chart flow
  (Walk Circle, Field Size MLC, Field Size Jaw)
- Batch report generator (`batch_generate_reports.py`)
- Picket Fence QA tab: auto-detect or manual load, per-leaf deviation analysis,
  portal image + deviation chart figure, PASS/FAIL/warning banner
- PF page in PDF (inserted between Field Size and Portal Images pages)
- PF result row in electronic signature block
- PF DICOM excluded from WL/field-size analysis when co-located in same folder
- Settings dialog: add/remove machines and physicists (persisted, live-refreshes
  the dropdowns), manual per-machine MLC/Jaw field-size reference entry
- App branding ("Daily QA - WL - DLG - Picket Fence") centralized in `APP_NAME`
  and applied across window title, header, dialogs, and PDF report/footer
- Portal-image panels window off a void-local ROI so the air void renders as
  a clearly visible dark circle instead of near-uniform gray
- Phantom ring (MIMI internal-ring) detection removed — was a secondary
  sanity check, not a PASS/FAIL criterion, and cluttered the portal panels

### Known behaviour / edge cases
- 156724 Jun 22 session in `WL Test Data/` has a missing G180 image — batch
  processor skips it with an error; this is a data gap, not a code bug.
- Bending magnet thermal drift on 153991 produced elevated Lat walk on Jun 22/23/29;
  these are real physics failures (beam steering), not phantom setup errors.
  The phantom is CBCT-corrected before acquisition and cannot move between shots.
- `avg_width` reported in the PF stats card is the measured 50%-penumbra radiation
  width (~12 mm), which is wider than the 10 mm nominal geometric gap due to MV
  beam penumbra; this is expected and does not affect the deviation metric.

### Pending / future ideas
- PF trend database (store per-day max deviation and RMS per machine)
- PF trend chart in the View Trends window
- Multi-picket fence support (if the department acquires multi-picket images)
