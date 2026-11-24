# Changelog

All notable changes to the Daily WL QA Tool are documented here.

---

## v2.0.0 - 2026-11-24

### Packaging
- **Self-contained Windows release** (`DailyQA-portable.zip`, built by GitHub Actions on every `v*` tag): embedded Python and every library, no installation or internet; `start-daily-qa.bat`, `backup-data.bat`, `README-portable.txt`. The build runs the tests, then starts the application window headless with the bundled Python before it zips.

### Added - bulk import and new trend analysis (2026-11-24)
- **Import all new sessions** (Check EPID Folder > list): analyses and saves every new complete session (Winston-Lutz record, field sizes, picket fence) without opening each one. One session at
  a time and each in its own try/except, so a bad session is reported and the run goes on; cancelable; resumable (saved sessions are registered and never analysed twice); image arrays are dropped as
  soon as a session is analysed; the log folder is read once per run. A session is saved only when it is complete and the void was found reliably at every angle; the others are listed with the reason
  (also written to `daily_import.log`) and left for review. Works with no logs, an empty or unreadable log folder, or logs that do not match: the session is then analysed without a log.
- **Trend analysis** (`daily_trends.py`) for each machine: (1) radiation isocenter radius and its drift (straight-line slope in mm / 30 days with its p-value, recent-vs-baseline median, raw SI
  offset at each gantry angle); (2) jaw size and (3) MLC size against the reference with the same statistics; (4) picket fence maximum and RMS leaf error, the worst leaf of every session, the
  end-to-end tilt of the leaf errors, and whether the SAME leaf is repeatedly the worst (binomial test corrected for picking the commonest of 40 leaves; per-leaf mean error in standard errors).
  Step changes (a calibration, a repair) are found and marked on the plots; lines are not drawn across gaps of more than 21 days. A drift is called only when it is statistically firm AND larger
  than 0.10 mm (isocenter), 0.15 mm (field size) or 0.05 mm (picket fence) per 90 days.
- Picket-fence results are now stored (`pf_records`, with every leaf's deviation) when a report is generated or a session is imported; `rebuild_history.py` fills them from the images (23 records on
  2026-11-24). The old trend dialog (dates as text on the x axis) is replaced.
- What the real data showed (2026-06 to 2026-11, 71 sessions): no isocenter drift on any machine (slopes within +/-0.012 mm / 30 days, all p > 0.1); on 2026-06-17/18 the MLC size (and on
  VersaTwo the jaw size) of all three machines stepped by +1.1 to +1.8 mm together (a common cause, most likely the test field, not the machines); the picket-fence leaf errors tilt by 0.2-0.3 mm along the
  picket on every machine (so an END leaf is always 'worst'); with the tilt removed no leaf repeats.

### Changed - grading basis (2026-11-24)
- **PASS/FAIL is the maximum displacement of any beam from the isocenter (the walk radius) against 1.0 mm** (TG-198 / MPPG: < 1 mm for SRS; all three linacs are used for SRS),
  not the sphere diameter. The diameter test failed 66 of the 71 stored sessions (1.0-1.6 mm diameters are the normal gantry sag of these machines); on the displacement basis
  all 71 pass (largest 0.82 mm). `max_2d_walk_mm` is now the radius (`walk_diameter_mm` stays as information), the tolerance ring is drawn at radius 1.0 mm, the per-angle
  colours use 0.5 / 1.0 mm, the trend plots the displacement, and the stored `pass_fail` of every record was re-derived from `walk_circle_r`
  (old file kept as `wl_qa_history.db.bak-before-displacement-20261124`). Change `TOLERANCE_MM` to change the limit (1.5 for SBRT, 2.0 general).

### Fixed (review 2026-11-24)
- A folder that failed to load (empty, wrong files) left the PREVIOUS session's results on screen under the new folder's name with the report button enabled, so they could be
  reported or saved as the new session. The results are now cleared and the button disabled.
- With two images at one gantry angle in a folder (a repeated delivery), the one analysed depended on the order the files were listed in; it is now the later by acquisition time
  (as the native-export path already did). The folder listing was also quadratic in the number of files.
- A folder holding several sessions (the `Daily WL V2 07012026` test folder has six: three machines, two days) was analysed as a mixture of whichever images were listed last. Images are now grouped into sessions
  (a different patient id or a pause of 30 minutes) and the newest session with all four angles is analysed; the folder label says so ("newest of 6 sessions in the folder").
- Generating the report twice for one session saved two records (two trend points); a record for the same machine and time is now replaced.

### Added
- **Native Elekta iView export** (`img_<uid>` folders with a lossless JPEG and `_Frame.xml`, no DICOM): read by `iview_native.py` for both *Load DICOM Directory* and
  *Check EPID Folder*. The images carry no patient id (taken from the log file names when logs are in the folder), no label (the WL and picket-fence images are
  recognised from their shape: 40 x 40 mm field with the void, one 20 cm picket) and no pixel spacing (0.4 mm at the panel = 0.25 mm at isocenter, which reads the
  40 mm field as 40.0 mm). Logs delivered in the imported folder are used when the machine has no log folder or nothing matches there.
- Log matching accepts the imager clock behind the machine's as well as ahead (it was +4.5 min one day, -7.7 min another).
- **Check EPID Folder**: per-machine EPID image folder (Settings > Data Folders). The button searches it for Daily WL sessions not processed yet and opens them
  (several: a list with a quick result for each; *Set aside* hides one). Manual "Load DICOM Directory" is unchanged.
- **Optional machine logs**: per-machine log folder; `.rllf` logs are paired with the images and the field centres corrected for the aperture the collimator
  actually had. Works without logs; *Use machine log* checkbox. The result and PDF state whether a log was used.

### Fixed
- **A saved record carried the time it was SAVED, not the time the images were acquired**, so trend dates moved whenever a session was saved later (30 of the 45 old records
  were stamped by a batch run). `_save_to_db` now takes the acquisition time (the PDF already did).
- **`rebuild_history.py`** rebuilds the history database from the images (every complete session under `WL Test Data` / `Test Data`, current methods, logs when present,
  acquisition date and time). Run on 2026-11-23: 71 sessions replaced the 45 old records (old file kept as `wl_qa_history.db.bak-before-rebuild-20261123`).
- **Field-edge detection**: edges come from strips clear of the void, thresholded against the local plateau; the 40 mm field now reads 40.6-40.7 mm (was ~43).
- **Void detection**: least-squares fit of the void shadow on a polynomial background replaces the minimum/centroid estimate, which was pulled toward the
  beam centre. A weak or missing void is flagged in the GUI instead of being measured silently.
- **Walk calculation used the wrong sign convention.** The panel turns with the gantry, so a static phantom offset has opposite signs at G0/G180 and
  at G90/G270; the old code subtracted the same offset at both angles and left the set-up error in the walk. The set-up offset is now fitted with
  `u = X cos G + Z sin G`, `v = Y`. On 66 stored sessions the largest radius falls from 1.54 mm to 0.86 mm.
- **The walk is now the exact smallest sphere** touching the four beam axes (replaces the 2D minimum enclosing circle with a free centre, which is not a
  physical quantity). Reproducible: equal minimisers resolve to the smallest offset.

### Changed
- **PASS/FAIL is on the isocenter diameter (twice the walk radius), limit 1.0 mm**, was radius. The figure, GUI, PDF, trend plot and report text show the
  diameter; the radius is kept in the database (`walk_circle_r`). Records saved before this change hold the old method's radius.

## [Unreleased]

### Changed
- **Walk circle Lateral/AP baseline now uses a minimax (Chebyshev) fit**
  instead of an independent per-axis-pair mean. `_fit_minimax_setup()` runs a
  2-parameter Nelder-Mead (`scipy.optimize`) directly against the exact
  Minimum Enclosing Circle radius, seeded at the old pair-mean values, so it
  is guaranteed to match or reduce (never inflate) the reported walk circle
  radius. SI baseline remains a plain mean (proven translation-invariant to
  the MEC radius). Validated against 62 real clinical WL sessions: minimax
  matched or improved every session with zero PASS/FAIL flips; most sessions
  were within 0.0004 mm of the old method, and one grossly out-of-tolerance
  session (~7.6 mm raw walk, a hardware/setup problem, not normal daily QA
  data) showed a real 0.11 mm reduction, confirming the pair-mean is not
  always optimal. See CLAUDE.md "Walk circle metric" section for the full
  rationale.

---

## 2026-07-03 — Single-import Picket Fence loading, per-machine report paths, PF avg_width fix

### Changed
- **Single DICOM import for WL, field size, and Picket Fence**: removed the
  separate **Load Picket Fence** button and its `_load_pf_directory` handler.
  `Load DICOM Directory` is now the only import path — drop the PF DICOM in
  the same daily folder as the 4 WL images and it's picked up automatically.
  Fixed the underlying bug that made this unreliable: the PF auto-detect scan
  only checked the top level of the selected folder (`Path.iterdir()`), while
  real iViewGT exports nest every RTIMAGE (including a co-located PF image)
  in its own UID-named subfolder. Both the main auto-detect loop and the
  (now-removed) manual loader's scan now recurse (`Path.rglob("*")`),
  matching how `load_dicom_images()` already scanned for the WL images.
- **PDF save location remembered per machine**: the "Save Daily QA Report"
  dialog now opens to `config["last_report_dir_by_machine"][machine]` instead
  of one shared `last_report_dir`, so different machines' reports can live in
  different folders without each save overwriting the other's default
  location. Falls back to the previous shared path (then home) for a machine
  with no saved history yet.
- **Picket Fence `avg_width` restricted to central ±5 cm**: the average
  measured leaf width shown in the PF stats card and PDF report (new
  `pf_avg_width_central_mm()` helper, `PF_AVG_WIDTH_HALF_RANGE_MM = 50.0`)
  now excludes leaves near the ends of the 200 mm picket, where MV beam
  penumbra widens the 50%-threshold width measurement and previously skewed
  the reported average.

---

## 2026-07-01 — Remove phantom ring detection; improve void visibility

### Removed
- **Phantom ring detection**: deleted `measure_phantom_ring()`, its constants
  (`RING_SEARCH_MIN_MM`, `RING_SEARCH_MAX_MM`, `RING_GAUSSIAN_SIGMA`,
  `RING_RADIUS_NOMINAL_MM`), the `ring_info` field on image results, the
  magenta dashed ring overlay on each portal panel, and the "Phantom Ring:"
  status label/refresh logic in the bottom status bar. It was a secondary
  sanity indicator (not a PASS/FAIL criterion) that cluttered the portal
  images without adding QA value beyond the existing void/field detection.

### Changed
- **Portal image windowing (diagnostic figure)**: the 4 portal panels now
  compute contrast windowing from a small ROI centred on the *detected void*
  (± 5× void radius, 1st–97th percentile of that ROI) instead of the whole
  field crop. Previously the void — a subtle local minimum riding on a much
  larger field→background intensity range — was windowed into near-uniform
  dark gray and effectively invisible. The new windowing saturates the field
  edge/background to black/white but renders the void as a clearly visible
  dark circle.

---

## 2026-07-01 — Settings dialog, split jaw tolerance, rebuilt trends, app rename

### Added
- **Settings dialog**: new "Settings" button opens a dialog with three tabs:
  - **Machines** / **Physicists** — add/remove entries directly in the app
    (`QListWidget` + Add/Remove buttons). Persisted to `wl_qa_config.json`;
    the main window's dropdowns refresh live and keep the current selection.
    At least one entry must remain in each list.
  - **Field Size Reference** — manual numeric MLC/Jaw reference entry per
    machine, supplementing the existing "Set Current as Reference" button
    (which derives the reference from a live measurement instead).
  - `MACHINES`/`PHYSICISTS` module constants are now only the factory
    defaults used on first run; the app reads/writes
    `config["machines"]`/`config["physicists"]` at runtime.

### Changed
- **Physical jaw field-size tolerance split from MLC**: jaw deviations now
  warn at **0.6 mm** and fail at **0.8 mm** (`FIELD_SIZE_JAW_WARN_MM` /
  `FIELD_SIZE_JAW_FAIL_MM`), independent of the existing MLC thresholds
  (0.4 mm warn / 0.6 mm fail). Updated across the GUI (Field Size QA tab,
  Results tab, PASS/WARN/FAIL banners) and the PDF report (banner, tables,
  methodology notes). Individual leaf-span deviations continue to use the
  MLC thresholds.
- **View Trends rebuilt**: replaced the single fixed walk-circle-only chart
  with a **machine picker → test picker → chart** flow. Pick a machine (only
  machines with existing records are listed), then a test — Walk Circle
  Radius, Field Size (MLC), or Field Size (Jaw) — and see that combination's
  trend chart and history table, colour-coded with the same thresholds used
  elsewhere in the app. "Change Test"/"Change Machine" buttons navigate back
  without reopening the dialog.
- **App renamed**: branding changed from "Winston-Lutz Daily QA" to
  **"Daily QA - WL - DLG - Picket Fence"** across the window title, header
  label, Settings/Trends dialog titles, and the PDF report title/footer
  (centralized in a new `APP_NAME` constant). "Winston-Lutz"/"WL" is kept in
  clinical/methodology text since that names the actual test, not the app.

---

## 2026-06-30 — Picket Fence display polish and bug fixes

### Changed
- **PF portal image windowing**: the figure now displays the normalised-inverted
  array (high dose = bright pixel) windowed from the central 90% of field rows
  (5th–95th percentile of that core region).  End-of-picket penumbra and the
  Elekta hot-spot artefact at the beam ends are excluded from the window
  calculation, eliminating the saturated white blowout seen previously.
- **PF thresholds tightened**: `PF_TOLERANCE_MM` 1.5 → **0.5 mm** (fail);
  new `PF_WARN_MM` = **0.4 mm** (warning).
- **PF deviation chart**: orange ±0.4 mm warning lines added alongside the red
  ±0.5 mm fail lines; bars and portal-image dots now follow a three-tier colour
  scheme (green < 0.4 mm / orange 0.4–0.5 mm / red ≥ 0.5 mm).  X-axis scale
  zoomed in — range = max(2 × fail tol, 1.5 × max dev, 0.3 mm).
- **PF leaf count capped at 40**: the 12%-threshold field-boundary detection sits
  in the penumbra, making the measured field height slightly taller than 200 mm;
  `round()` produced 41 leaves.  Count is now capped at `PF_LEAVES_TOTAL // 2`
  (40 leaf pairs).
- **Stats card Max |Δ| colour**: follows the same green / orange / red rule as
  the chart.

### Fixed
- **PF file excluded from WL/field-size analysis**: when the Picket Fence DICOM
  is placed in the same directory as the four WL images, `load_dicom_images()`
  now calls `identify_pf_dicom()` and skips it before gantry-angle matching.
  Previously the PF image (acquired at G0) was displacing the real G0 WL image
  and causing the field-size analysis to fail.

### Added
- **PF result row in PDF signature**: the electronic signature table on the final
  page gains a "PF Result" row (PASS/FAIL, Max |Δ|, leaf count, tolerances)
  whenever PF data is included in the report; background follows the green /
  amber / red threshold scheme.

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
