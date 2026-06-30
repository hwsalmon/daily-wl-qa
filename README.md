# Daily WL QA

**Daily Winston-Lutz isocenter walk QA for Elekta Versa HD LINACs**
using the Standard Imaging MIMI Phantom (6.4 mm air void).

![App Icon](icon.png)

---

## What it does

Loads four Elekta iViewGT portal DICOM images (G0 / G90 / G180 / G270),
automatically detects the MIMI phantom air void in each image, computes the
**radiation/mechanical walk circle** via the Minimum Enclosing Circle (MEC)
algorithm, and generates a signed PDF QA report — in seconds.

| Feature | Detail |
|---------|--------|
| **Input** | Elekta iViewGT RTIMAGE DICOM files (4 cardinal angles) |
| **Phantom** | Standard Imaging MIMI — 6.4 mm air void |
| **Algorithm** | MEC of all 4 corrected walk residuals |
| **Pass/Fail** | Walk circle radius ≤ 1.0 mm |
| **Machine detection** | Auto-selects machine from DICOM `PatientID` tag |
| **Report date** | Always uses DICOM `StudyDate`/`StudyTime` — never today's date |
| **Picket Fence QA** | MLC leaf position accuracy — auto-detected or manually loaded |
| **Output** | 3–4 page PDF report + SQLite trend database |
| **Batch processing** | `batch_generate_reports.py` — all sessions, all machines |
| **Platforms** | Linux · Windows · macOS |

---

## Installation

### Windows (hospital / restricted PC) — no admin rights needed

> No system Python required. The setup script downloads its own
> self-contained Python runtime into the app folder.

**Step 1** — Download the app

Click **Code → Download ZIP** on GitHub, unzip anywhere (e.g. your Desktop or `Documents`).

**Step 2** — Run setup (one time only)

Double-click **`setup_windows.bat`**

This will:
- Download Python 3.12 embeddable (~8 MB) into a `python_runtime` folder inside the app
- Install all dependencies (PySide6, pydicom, scipy, reportlab, matplotlib, Pillow, OpenCV)
- Generate the app icon

**Step 3** — Launch the app

Double-click **`run_wl_qa.bat`**

That's it. No Python installation, no admin rights, no IT involvement.

> **Already have rad-inventory installed?**  The setup is identical — it uses the same
> self-contained embedded Python approach.  You can run `setup_windows.bat` from the
> WL QA folder independently; it will not affect rad-inventory.

---

### Linux / macOS

```bash
git clone https://github.com/hwsalmon/daily-wl-qa.git
cd daily-wl-qa
python3 install.py
```

The installer handles dependencies, icon generation, and adds *Daily WL QA* to
your application menu automatically.

---

## Manual install (fallback)

```bash
pip install -r requirements.txt
python3 wl_qa_tool.py          # Linux / macOS
python  wl_qa_tool.py          # Windows (if Python is on your PATH)
```

---

## System requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.10 or newer |
| RAM | 512 MB |
| Disk | 400 MB (including dependencies) |
| Display | 1280 × 800 or larger |
| OS | Linux (X11 or Xwayland), Windows 10/11, macOS 12+ |

### Python packages installed automatically

| Package | Purpose |
|---------|---------|
| `PySide6` | Qt-based GUI framework |
| `pydicom` | DICOM file reading |
| `opencv-python` | Image processing |
| `scipy` | Gaussian filter, blob analysis |
| `reportlab` | PDF generation |
| `matplotlib` | Diagnostic figure rendering |
| `Pillow` | Image scaling |

---

## Usage

### 1. Select machine and physicist
Use the dropdowns at the top of the window before loading data.

### 2. Load DICOM directory
Click **Load DICOM Directory** and select the folder containing the four
Elekta iViewGT portal DICOM files from that day's WL acquisition.
The folder should contain one file per cardinal gantry angle
(G0, G90, G180, G270 — within ±5° of each cardinal).

The machine dropdown **auto-selects** the correct unit based on the `PatientID`
DICOM tag — no manual selection needed for the three configured Versa HD units.

### 3. Review results
- **PASS/FAIL banner** — walk circle radius vs. 1.0 mm tolerance, shown immediately.
- **Results tab** — raw displacements (Field → Void) and corrected displacements
  (3D setup error removed) for all four angles, colour-coded green/amber/red.
  The corrected-table subtitle shows the live **3D CBCT setup error**
  (Lateral / SI / AP mm) derived from all four images.
- **Portal Images tab** — 5-panel diagnostic figure: one portal image per angle
  (field boundary overlay, crosshair, void marker, displacement arrow, scale bar)
  plus a 2-D displacement map showing the walk circle.

### 4. Generate report
Click **Generate Daily Report (PDF)**.

The report is a consistent **3-page layout**:

**Page 1 — WL results**
- Metadata table: date/time from DICOM header, machine, physicist, phantom,
  tolerance, **3D CBCT setup error** (Lateral / SI / AP mm)
- Colour PASS/FAIL banner with walk circle radius
- Corrected displacements table + walk circle figure (side-by-side)
- Winston-Lutz methodology notes

**Page 2 — Field size results**
- Field size PASS/FAIL banner
- Angle displacement table + leaf span table (side-by-side)
- Field size methodology notes

**Page 3 — Portal images & signature**
- 5-panel diagnostic figure (4 portal images + 2-D displacement map)
- **Electronic signature block** (physicist, study date/time, 21 CFR Part 11 statement)

The report date, filename, and PDF internal metadata (`CreationDate`, `ModDate`)
all use the DICOM `StudyDate`/`StudyTime` — never the generation date.

Each report generation automatically saves a record to `wl_qa_history.db`.

### 5. Picket Fence QA (optional — performed weekly or as scheduled)

If a PF DICOM is present in the same directory as the WL images, it is
auto-detected and analysed automatically.  To load PF data from a separate
folder, click **Load Picket Fence**.

The **Picket Fence** tab shows:
- A portal image with a per-leaf measurement overlay
- A horizontal bar chart of per-leaf centre deviations (green / orange / red)
- Summary stats: N leaves, Max |Δ|, RMS, average measured width

A third PASS/FAIL banner (green/red/amber) appears alongside the WL and Field
Size banners only on days when PF data is loaded.

**Thresholds** (editable via constants at the top of `wl_qa_tool.py`):

| Level | Threshold | Colour |
|-------|-----------|--------|
| Pass | < ±0.4 mm | Green |
| Warning | 0.4 – 0.5 mm | Amber |
| Fail | ≥ ±0.5 mm | Red |

When the Picket Fence DICOM is placed in the **same directory** as the four WL
images it is automatically excluded from the WL and field-size analysis (the
`GantryAngle = 0°` of the PF image would otherwise displace the real G0 WL
image).

When the PDF report is generated on a PF day, it includes an extra page with
the PF results between the Field Size and Portal Images pages (4 pages total),
plus a **PF Result row** in the electronic signature block.
On non-PF days the report remains 3 pages.

### 6. View trend analysis
Click **View Trends** at any time to see a per-machine time-series plot of
walk circle radius with tolerance and advisory lines, plus a scrollable table
of all historical records.

---

## Machine and physicist configuration

Machines, physicists, and the PatientID → machine mapping are defined near the
top of `wl_qa_tool.py` (lines ~115–135):

```python
MACHINES = [
    "Elekta VersaHD 153991",
    "Elekta VersaHD 156724",
    "Elekta VersaHD 154613",
]

PHYSICISTS = [
    "Howard W. Salmon, PhD, DABR",
    "Shawn Hollars, MS, DABR",
    "Logen Hall, MS, DABR",
]

# Maps Elekta iViewGT PatientID → machine name for auto-detection
PATIENT_ID_MACHINE_MAP = {
    "QA_Daily_V1_26": "Elekta VersaHD 153991",
    "QA_Daily_V2_26": "Elekta VersaHD 156724",
    "QA_Daily_MV_26": "Elekta VersaHD 154613",
}
```

To add or change machines or physicists, edit those three structures and save —
no other changes are needed.  When a loaded DICOM directory contains a recognised
`PatientID`, the machine dropdown selects automatically.

---

## Batch processing

`batch_generate_reports.py` processes all sessions in `WL Test Data/` in one run:

```bash
python3 batch_generate_reports.py
```

It scans for session directories, identifies each machine via `PatientID`, loads
and analyses the DICOM images, generates PDF reports (with study-date filenames
and metadata), and saves all records to `wl_qa_history.db`.  Physicist assignment
is controlled by the date range defined at the top of the script.

---

## DICOM requirements

- Elekta iViewGT RTIMAGE files (modality `RTIMAGE`)
- Files **do not** need a DICOM File Meta Information header (the Elekta
  iViewGT format omits the Transfer Syntax UID — handled automatically)
- One file per gantry angle; the `GantryAngle` DICOM tag is used to assign
  each file to its nearest cardinal angle
- Pixel spacing and SID/SAD are read from the DICOM header for automatic
  magnification correction

---

## Clinical notes

### Portal coordinate system

Elekta iViewGT stores portal images with a **consistent patient orientation** at
all gantry angles (no image flip at opposing gantry positions).

| Gantry | Portal X | Portal Y |
|--------|----------|----------|
| G0° / G180° | Patient **Lateral** (L/R) | Patient **SI** (S/I) |
| G90° / G270° | Patient **AP** (A/P) | Patient **SI** (S/I) |

The 3D CBCT residual setup error is therefore:

```
Lateral = (ΔX_G0  + ΔX_G180) / 2
SI      = mean(ΔY_G0, ΔY_G90, ΔY_G180, ΔY_G270)
AP      = (ΔX_G90 + ΔX_G270) / 2
```

Subtracting the angle-appropriate component from each raw displacement yields
pure mechanical walk residuals. The **Minimum Enclosing Circle** radius of those
four residual vectors is the walk circle metric.

### Pass/Fail criterion

Walk circle radius ≤ **1.0 mm** (editable via `TOLERANCE_MM` in `wl_qa_tool.py`).

### Void detection

The MIMI air void receives *more* dose than the surrounding acetal phantom
(air attenuates less at MV energies). Elekta iViewGT stores pixels inverted
(LOW value = HIGH dose), so the void is the **local minimum** within the
irradiated field. Detection pipeline:

1. Gaussian pre-filter (σ ≈ void_radius × 0.30 px) — suppresses MV noise
2. Global minimum pixel within the central 50% of the field
3. Inverse-intensity-squared centroid — sub-pixel void localisation

---

## File layout

```
wl_qa_tool.py              Main application (single file — no build step)
batch_generate_reports.py  Batch PDF generator for multiple sessions/machines
install.py                 One-step Linux/macOS installer
run_wl_qa.bat              Windows double-click launcher
setup_windows.bat          Windows first-time setup (embeddable Python)
requirements.txt           Python dependency list
icon.png                   App icon (256 px)
icon_512.png               App icon (512 px master)
icon.ico                   Windows multi-resolution icon (16–256 px)
CLAUDE.md                  Developer / AI assistant reference
wl_qa_history.db           SQLite trend database (auto-created, not in git)
wl_qa_config.json          Last-used paths (auto-created, not in git)
```

---

## Trend database

`wl_qa_history.db` is created automatically in the same folder as
`wl_qa_tool.py` on first run.  It is a standard SQLite file and can be
opened with any SQLite browser (e.g. [DB Browser for SQLite](https://sqlitebrowser.org/)).

The database is excluded from git (`.gitignore`) because it contains
site-specific QA records.

---

## License

For clinical and research use at Franciscan Health Indianapolis.  
Contact: Howard W. Salmon, PhD, DABR — howard.w.salmon@gmail.com
