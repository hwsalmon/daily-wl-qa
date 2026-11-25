#!/usr/bin/env python3
"""
Daily QA - WL - DLG - Picket Fence — Elekta Versa HD / Standard Imaging MIMI Phantom

Air void detection: the MIMI phantom uses a 6.4 mm diameter AIR VOID as the
isocenter target. Air attenuates less than the surrounding acetal copolymer
(~1.41 g/cm³) so the void receives more dose than the phantom body. Elekta
iViewGT RTIMAGEs are stored with INVERTED pixel values (LOW value = HIGH dose),
so the void is the LOCAL MINIMUM within the irradiated field — detected as the
darkest spot in the field-windowed portal image.

Baseline correction: the G0° displacement (residual CBCT couch setup error) is
subtracted from all cardinal-angle displacements to isolate true isocenter walk.
"""

import os
import sys
import shutil
import sqlite3
import datetime
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Dependency checks ─────────────────────────────────────────────────────────
_missing = []
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QComboBox, QTabWidget, QFrame, QDialog, QScrollArea,
        QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy,
        QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem,
        QHeaderView, QTreeWidget, QTreeWidgetItem, QDoubleSpinBox, QListWidget, QListWidgetItem,
        QLineEdit, QInputDialog, QStackedWidget, QCheckBox, QProgressDialog, QFormLayout, QGroupBox, QDateEdit, QTextEdit,
    )
    from PySide6.QtCore import Qt, QDate, QUrl
    from PySide6.QtGui import QPixmap, QIcon, QColor, QBrush, QDesktopServices, QFont
except ImportError:
    _missing.append("PySide6")

try:
    import pydicom
except ImportError:
    _missing.append("pydicom")

try:
    import cv2
except ImportError:
    _missing.append("opencv-python")

try:
    from scipy.ndimage import gaussian_filter, gaussian_filter1d, label as scipy_label, sum as ndimage_sum
    from scipy.optimize import minimize as _scipy_minimize
except ImportError:
    _missing.append("scipy")

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle,
        Paragraph, Spacer, HRFlowable,
        Image as RLImage,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
except ImportError:
    _missing.append("reportlab")

# matplotlib is imported with the Agg (non-interactive) backend so figure
# generation works whether or not a display is available.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

if _missing:
    print("Missing required packages. Install with:")
    print(f"  pip install {' '.join(_missing)}")
    sys.exit(1)

# ── Clinical constants ────────────────────────────────────────────────────────
TOLERANCE_MM        = 1.0    # Maximum DISPLACEMENT of any beam from the isocenter (PASS/FAIL): TG-198 / MPPG, < 1 mm for SRS (all three linacs are used for SRS)
VOID_DIAMETER_MM    = 6.4    # MIMI phantom air void diameter
FIELD_SIZE_MM       = 40.0   # Nominal field size (40×40 mm)
GANTRY_ANGLES       = [0, 90, 180, 270]
CARDINAL_TOLERANCE  = 5.0    # Degrees: how close to a cardinal to accept a file

# Void search region: fraction of field half-width from field center
# → 0.50 × 20 mm = 10 mm search radius (void should be well within this)
VOID_SEARCH_HALF_FIELD_FRACTION = 0.50

# Gaussian pre-filter sigma as a fraction of void radius in pixels
# (tuned for ~1–2 mm blur, removes MV portal image noise without eroding the void)
GAUSSIAN_SIGMA_FRACTION = 0.30

# Percentile threshold to isolate the void blob within the search crop
# The air void is among the very brightest pixels in the irradiated field
VOID_PERCENTILE_THRESHOLD = 85

# Acceptable void blob area range relative to ideal circle area
VOID_AREA_MIN_RATIO = 0.15
VOID_AREA_MAX_RATIO = 4.0

# ── Field size constants ───────────────────────────────────────────────────────
FIELD_SIZE_REF_MM   = 40.0                              # Nominal total field size (both axes)
MLC_LEAVES_PER_BANK = 8                                 # Leaves per MLC bank
MLC_LEAF_WIDTH_MM   = FIELD_SIZE_MM / MLC_LEAVES_PER_BANK  # 5.0 mm per leaf (SI)
FIELD_SIZE_WARN_MM  = 0.4   # MLC |deviation| > this → orange (warning)
FIELD_SIZE_FAIL_MM  = 0.6   # MLC |deviation| > this → red   (call service)
FIELD_SIZE_JAW_WARN_MM = 0.6   # Physical jaw |deviation| > this → orange (warning)
FIELD_SIZE_JAW_FAIL_MM = 0.8   # Physical jaw |deviation| > this → red   (call service)

MACHINE_NAME = "Elekta Versa HD"
PHANTOM_NAME = "Standard Imaging MIMI"

APP_NAME = "Daily QA - WL - DLG - Picket Fence"
APP_VERSION = "2.2.0"

MACHINES = [
    "Elekta VersaHD 153991",
    "Elekta VersaHD 156724",
    "Elekta VersaHD 154613",
]

# iViewGT PatientID → machine name (configured in MOSAIQ per linac)
PATIENT_ID_MACHINE_MAP = {
    "QA_Daily_V1_26": "Elekta VersaHD 153991",
    "QA_Daily_V2_26": "Elekta VersaHD 156724",
    "QA_Daily_MV_26": "Elekta VersaHD 154613",
}

PHYSICISTS = [
    "Howard W. Salmon, PhD, DABR",
    "Shawn Hollars, MS, DABR",
    "Logen Hall, MS, DABR",
]

# ── Picket Fence constants ─────────────────────────────────────────────────────
PF_TOLERANCE_MM  = 0.5   # Per-leaf centre deviation FAIL threshold (mm)
PF_WARN_MM       = 0.4   # Per-leaf centre deviation WARNING threshold (mm)
PF_LEAF_WIDTH_MM = 5.0   # Agility inner MLC leaf width at isocenter (mm)
PF_LEAVES_TOTAL  = 80    # Central leaves tested: 40 per bank
PF_AVG_WIDTH_HALF_RANGE_MM = 50.0  # avg_width stat: central +/- range only (excludes end penumbra)

DB_PATH     = Path(__file__).parent / "wl_qa_history.db"
APP_DIR     = Path(__file__).parent
CONFIG_PATH = Path(__file__).parent / "wl_qa_config.json"


def _load_config() -> dict:
    import json
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def _save_config(data: dict) -> None:
    import json
    try:
        CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


# ── DICOM loading ─────────────────────────────────────────────────────────────

def read_image(path):
    """One EPID image: a pydicom dataset, or - for an iView `img_<uid>` folder (native export: lossless JPEG + _Frame.xml) - the adapter that looks like one."""
    if Path(path).is_dir():
        import iview_native
        return iview_native.NativeDS(Path(path))
    return pydicom.dcmread(str(path), stop_before_pixels=False, force=True)


def _native_patient(root: Path) -> str:
    """The patient id from the machine logs delivered with a native export (QA_Daily_V1_26_...rllf), '' when there are none: the images carry no patient id."""
    try:
        import wl_sources
        for p in sorted(Path(root).rglob("*.rllf")):
            m = wl_sources.NAME_RE.match(p.name)
            if m and m["patient"] in PATIENT_ID_MACHINE_MAP:
                return m["patient"]
    except Exception:
        pass
    return ""


def _load_native_images(root: Path) -> dict:
    """{cardinal angle: NativeDS} for a native iView export (latest image at each angle), or {} when the folder holds none."""
    try:
        import iview_native
    except Exception:
        return {}
    if not iview_native.is_native_folder(root):
        return {}
    wl = [d for d in iview_native.scan_native(root, APP_DIR) if d.kind == "wl"]
    if not wl:
        return {}
    pid = _native_patient(root)
    images = {}
    for ds in wl:                                                   # oldest first: a repeated delivery replaces the earlier one
        g = float(ds.GantryAngle)
        cardinal = min(GANTRY_ANGLES, key=lambda a: min(abs(a - g), abs(360 + a - g), abs(a - g - 360)))
        if min(abs(cardinal - g), abs(360 + cardinal - g), abs(cardinal - g - 360)) > CARDINAL_TOLERANCE:
            continue
        ds.PatientID = pid
        ds.RTImageLabel = f"WL{len(images) + 1}"
        images[cardinal] = ds
    missing = [a for a in GANTRY_ANGLES if a not in images]
    if missing:
        raise ValueError(f"Missing images for gantry angles: {missing}\nFound angles for: {sorted(images.keys())}")
    return images


def load_dicom_images(directory: str) -> dict:
    """
    Scan directory (recursively) for DICOM files, map each to the nearest cardinal
    gantry angle.  Returns {0: ds, 90: ds, 180: ds, 270: ds}.
    Raises ValueError if any cardinal angle is missing.

    Uses force=True to handle Elekta RTIMAGE files that lack a standard DICOM
    File Meta Information header (common on older Elekta iViewGT exports).
    """
    root = Path(directory)
    native = _load_native_images(root)
    if native:
        return native
    # Every file in the tree named *.dcm / *.DCM / *.dicom, or a bare UID file (one pass over the tree)
    dcm_files = [f for f in sorted(root.rglob("*")) if f.is_file() and f.suffix.lower() in (".dcm", ".dicom", "")]

    if not dcm_files:
        raise ValueError(f"No files found in:\n{directory}")

    images = {}
    skipped = []
    found = []                                  # (acquisition time, cardinal angle, dataset): a repeated delivery at an angle is decided by TIME, not by the order files are listed in
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False, force=True)
            if identify_pf_dicom(ds):
                skipped.append(f"{f.name} (PF image — excluded from WL analysis)")
                continue
            gantry = float(getattr(ds, "GantryAngle", -999))
            if gantry < 0:
                skipped.append(f.name)
                continue
            # Angle difference accounting for 0/360 wrap
            cardinal = min(
                GANTRY_ANGLES,
                key=lambda a: min(
                    abs(a - gantry),
                    abs(360 + a - gantry),
                    abs(a - gantry - 360),
                ),
            )
            diff = min(
                abs(cardinal - gantry),
                abs(360 + cardinal - gantry),
                abs(cardinal - gantry - 360),
            )
            if diff > CARDINAL_TOLERANCE:
                skipped.append(f"{f.name} (G={gantry:.1f}°)")
                continue
            when = ""
            for dtag, ttag in (("ContentDate", "ContentTime"), ("AcquisitionDate", "AcquisitionTime"), ("StudyDate", "StudyTime")):
                d_, t_ = str(getattr(ds, dtag, "") or ""), str(getattr(ds, ttag, "") or "")
                if len(d_) == 8:
                    when = d_ + t_.split(".")[0].zfill(6)[:6]
                    break
            found.append((when, cardinal, ds, str(getattr(ds, "PatientID", "") or "")))
        except Exception as e:
            skipped.append(f"{f.name} ({e})")
    # A folder can hold several sessions (other machines, other days): group the images into sessions (a new patient id or a pause of 30 min) and analyse the NEWEST session that has all
    # four angles, so images of different machines or days are never mixed. Within a session the later of two images at one angle wins.
    found.sort(key=lambda x: x[0])
    sessions = []
    for item in found:
        last = sessions[-1][-1] if sessions else None
        gap = None
        if last is not None and last[0] and item[0]:
            gap = (datetime.datetime.strptime(item[0], "%Y%m%d%H%M%S") - datetime.datetime.strptime(last[0], "%Y%m%d%H%M%S")).total_seconds() / 60.0
        if last is None or (gap is not None and gap > 30.0) or (last[3] and item[3] and last[3] != item[3]):
            sessions.append([])
        sessions[-1].append(item)
    load_dicom_images.note = ""
    pick = next((ss for ss in reversed(sessions) if {c for _w, c, _d, _p in ss} >= set(GANTRY_ANGLES)), None)
    chosen = pick if pick is not None else found
    if pick is not None and len(sessions) > 1:
        load_dicom_images.note = f"newest of {len(sessions)} sessions in the folder ({pick[0][0][:4]}-{pick[0][0][4:6]}-{pick[0][0][6:8]} {pick[0][0][8:10]}:{pick[0][0][10:12]})"
    for _when, cardinal, ds, _pid in chosen:
        images[cardinal] = ds

    if skipped:
        print(f"Skipped {len(skipped)} file(s): {', '.join(skipped)}")

    missing = [a for a in GANTRY_ANGLES if a not in images]
    if missing:
        raise ValueError(
            f"Missing DICOM images for gantry angles: {missing}\n"
            f"Found angles for: {sorted(images.keys())}"
        )
    return images


def get_pixel_array(ds) -> tuple:
    """
    Extract calibrated float32 pixel array and the effective pixel spacing at
    isocenter (mm/pixel) from a DICOM RTIMAGE dataset.

    Pixel spacing in DICOM is reported at the DETECTOR plane.  To express
    displacements in mm at isocenter we apply the SAD/SID magnification:
        spacing_iso = spacing_detector × (SAD / SID)

    Elekta iViewGT RTIMAGE exports often lack a Transfer Syntax UID in the file
    meta header, causing pydicom 3.x to refuse pixel decoding.  We first attempt
    the normal path; on failure we inject ExplicitVRLittleEndian and retry; if
    that still fails we decode the raw PixelData buffer directly (valid for all
    uncompressed Little-Endian RTIMAGE files).
    """
    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        from pydicom.uid import ExplicitVRLittleEndian
        if not hasattr(ds, "file_meta") or ds.file_meta is None:
            ds.file_meta = pydicom.Dataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            arr = None

    if arr is None:
        bits   = int(getattr(ds, "BitsAllocated", 16))
        signed = int(getattr(ds, "PixelRepresentation", 0)) == 1
        dtype  = (np.int16 if signed else np.uint16) if bits == 16 else np.uint8
        raw    = np.frombuffer(bytes(ds.PixelData), dtype=dtype)
        arr    = raw.reshape(int(ds.Rows), int(ds.Columns)).astype(np.float32)

    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Detector-plane pixel spacing
    spacing_det = None
    for attr in ("ImagePlanePixelSpacing", "PixelSpacing", "ImagerPixelSpacing"):
        val = getattr(ds, attr, None)
        if val is not None:
            spacing_det = float(val[0])
            break
    if spacing_det is None:
        spacing_det = 0.392
        print("Warning: pixel spacing tag not found; defaulting to 0.392 mm/pixel")

    # Magnification correction: convert detector spacing → isocenter spacing
    sad = float(getattr(ds, "RadiationMachineSAD", 1000.0))
    sid = float(getattr(ds, "RTImageSID",          0.0))
    if sid > 0:
        spacing_iso = spacing_det * (sad / sid)
    else:
        spacing_iso = spacing_det
        print("Warning: RTImageSID not found; no magnification correction applied")

    return arr, spacing_iso


# ── Image analysis ────────────────────────────────────────────────────────────

def find_field_center(arr: np.ndarray, spacing_mm: float) -> tuple:
    """
    Locate the radiation field center from an Elekta iViewGT RTIMAGE.

    Storage convention: LOW pixel value = high dose (inverted relative to dose).
    The irradiated field is therefore the DARK region of the image against a
    near-saturated (bright) background.  We threshold at the 50th percentile of
    the normalised range, isolate the largest dark blob, and return two estimates:
      - BBox midpoint: (min+max)/2 per axis — 0.5-px quantisation, robust to
        interior inhomogeneity
      - Geometric centroid: mean of all field pixel coordinates — sub-pixel
        precision, sensitive to edge asymmetry

    Returns ((bbox_col, bbox_row), (centroid_col, centroid_row)).
    """
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    # Field = DARK (low norm); background = BRIGHT (high norm)
    field_mask = (norm < 0.50).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, k)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN,  k)

    labeled, n = scipy_label(field_mask > 0)
    if n == 0:
        cx = arr.shape[1] / 2.0
        cy = arr.shape[0] / 2.0
        return (cx, cy), (cx, cy)

    sizes = ndimage_sum(field_mask > 0, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    rows, cols = np.where(labeled == largest)

    bbox_col = (cols.min() + cols.max()) / 2.0
    bbox_row = (rows.min() + rows.max()) / 2.0
    cent_col = float(cols.mean())
    cent_row = float(rows.mean())
    return (bbox_col, bbox_row), (cent_col, cent_row)


def _find_void_minimum(
    arr: np.ndarray,
    spacing_mm: float,
    field_center_px: tuple,
) -> tuple:
    """
    Locate the 6.4 mm air void within the MIMI phantom near the radiation field center.

    Image convention (Elekta iViewGT RTIMAGE): LOW pixel value = high beam fluence.
    The air void (density ≈ 0) attenuates LESS than the surrounding acetal copolymer
    (density ≈ 1.41 g/cm³), so it transmits more beam and appears as a LOCAL MINIMUM
    in pixel value — the darkest region within the irradiated area.

    Algorithm:
      1. Crop a ± VOID_SEARCH_HALF_FIELD_FRACTION × field_half-width window.
      2. Gaussian-smooth (σ ≈ void_radius × GAUSSIAN_SIGMA_FRACTION px) to
         suppress MV portal noise.
      3. Locate the global minimum inside the smoothed crop, excluding a border
         strip of ≈ void_radius px to avoid penumbra contamination.
      4. Within a refinement window of radius ≈ 1.5 × void_radius centred on that
         minimum, compute an inverse-intensity-squared centroid for sub-pixel
         localisation (darker pixel = higher weight).

    spacing_mm is already magnification-corrected to isocenter (SAD/SID applied).

    Returns (col_px, row_px).
    """
    fx, fy = field_center_px
    nrows, ncols = arr.shape

    void_radius_px   = (VOID_DIAMETER_MM / 2.0) / spacing_mm
    search_radius_px = int(
        (FIELD_SIZE_MM * VOID_SEARCH_HALF_FIELD_FRACTION) / spacing_mm
    )

    r0 = max(0,     int(fy) - search_radius_px)
    r1 = min(nrows, int(fy) + search_radius_px)
    c0 = max(0,     int(fx) - search_radius_px)
    c1 = min(ncols, int(fx) + search_radius_px)

    crop     = arr[r0:r1, c0:c1].copy()
    sigma    = max(1.0, void_radius_px * GAUSSIAN_SIGMA_FRACTION)
    smoothed = gaussian_filter(crop, sigma=sigma)

    # ── Step 3: find the global minimum, avoiding penumbra at crop border ──
    border  = max(3, int(void_radius_px))
    ch, cw  = smoothed.shape
    if ch <= 2 * border or cw <= 2 * border:
        border = 0  # crop too small; skip border exclusion
    interior     = smoothed[border:ch-border, border:cw-border]
    min_flat_idx = np.argmin(interior)
    mr_int, mc_int = np.unravel_index(min_flat_idx, interior.shape)
    # Translate back to crop coordinates
    min_r = mr_int + border
    min_c = mc_int + border

    # ── Step 4: sub-pixel centroid in a local window around the minimum ────
    win   = max(5, int(void_radius_px * 1.5))
    wr0   = max(0,  min_r - win)
    wr1   = min(ch, min_r + win + 1)
    wc0   = max(0,  min_c - win)
    wc1   = min(cw, min_c + win + 1)

    wnd      = smoothed[wr0:wr1, wc0:wc1]
    # Invert and square: lower original value → much larger weight
    wnd_inv  = (wnd.max() - wnd) ** 2
    total    = wnd_inv.sum()

    if total > 1e-12:
        row_idx = np.arange(wr0, wr1, dtype=float)[:, np.newaxis]
        col_idx = np.arange(wc0, wc1, dtype=float)[np.newaxis, :]
        void_row_crop = float((row_idx * wnd_inv).sum() / total)
        void_col_crop = float((col_idx * wnd_inv).sum() / total)
    else:
        void_row_crop = float(min_r)
        void_col_crop = float(min_c)

    return c0 + void_col_crop, r0 + void_row_crop


# Empirical radial profile of the void shadow (fraction of its depth) every 0.253 mm from the centre, measured on 280 real MIMI images.
# It is wider than the 6.4 mm void itself because of scatter and detector blur; fitting it removes the sloping beam profile from the estimate.
_VOID_PROFILE = np.array([1.031, 1.03, 1.027, 1.022, 1.016, 1.006, 0.994, 0.978, 0.959, 0.932, 0.9, 0.867, 0.831, 0.792, 0.747, 0.7, 0.643, 0.577, 0.506, 0.433, 0.357, 0.268, 0.166, 0.068, 0.0])
_VOID_PROFILE_STEP_MM = 0.253
VOID_FIT_HALF_MM = 16.0        # fit window half-size
VOID_BG_ORDER = 4              # polynomial order of the background under the void
VOID_BG_RADIUS_MM = 9.0        # pixels beyond this radius define the first background estimate
VOID_DEPTH_RANGE = (600.0, 4000.0)   # plausible dip depth in counts (real images: 1100-1600); outside this the fit is not a void
VOID_MIN_SNR = 1.0             # dip depth / residual rms (the residual includes phantom structure the profile does not model)
VOID_MAX_SHIFT_MM = 6.0        # fit may not move further than this from the coarse start


def fit_void(arr: np.ndarray, spacing_mm: float, field_center_px: tuple) -> dict:
    """
    Void centre by least squares.  The coarse start is the minimum of the smoothed image (_find_void_minimum).  The refinement fits
        pixel = quadratic background(x, y)  -  depth * profile(|x - centre|)
    over a +/-16 mm window, so the slope of the beam profile (the 'bowl') no longer drags the estimate toward the beam centre
    (the old minimum / centroid estimate was pulled ~20 % of the offset toward the field centre in a test with a known void position).

    Returns {"col", "row", "depth", "noise", "snr", "ok", "method"}.  ok is False (and the coarse minimum is returned) when the fit cannot be
    trusted: an implausible or weak dip (depth outside VOID_DEPTH_RANGE or depth/residual < VOID_MIN_SNR, e.g. no phantom or no void in the image), the window leaves the image, or the fit wanders.
    """
    from scipy.optimize import least_squares

    c0 = _find_void_minimum(arr, spacing_mm, field_center_px)
    out = {"col": float(c0[0]), "row": float(c0[1]), "depth": 0.0, "noise": 0.0, "snr": 0.0, "ok": False, "method": "minimum"}
    h = int(round(VOID_FIT_HALF_MM / spacing_mm))
    x0, y0 = int(round(c0[0])), int(round(c0[1]))
    if y0 - h < 0 or x0 - h < 0 or y0 + h + 1 > arr.shape[0] or x0 + h + 1 > arr.shape[1]:
        return out
    crop = arr[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1].astype(np.float64)
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float64)
    X, Y = xx * spacing_mm, yy * spacing_mm
    grid = np.arange(len(_VOID_PROFILE)) * _VOID_PROFILE_STEP_MM

    terms = [(i, j) for i in range(VOID_BG_ORDER + 1) for j in range(VOID_BG_ORDER + 1 - i)]
    A = np.stack([(X / 10.0) ** i * (Y / 10.0) ** j for i, j in terms], -1)

    def model(p):
        r = np.hypot(X - p[0], Y - p[1])
        return A @ p[3:] - p[2] * np.interp(r, grid, _VOID_PROFILE, right=0.0)

    cx, cy = (c0[0] - x0) * spacing_mm, (c0[1] - y0) * spacing_mm
    m = np.hypot(X - cx, Y - cy) > VOID_BG_RADIUS_MM
    coef = np.linalg.lstsq(A[m], crop[m], rcond=None)[0]
    d0 = float((A @ coef - crop)[np.hypot(X - cx, Y - cy) < 2.0].mean())
    p0 = np.r_[cx, cy, max(d0, 500.0), coef]
    try:
        r = least_squares(lambda p: (model(p) - crop).ravel(), p0,
                          x_scale=np.r_[1, 1, 1000, np.full(len(terms), 1000.0)], max_nfev=200)
    except Exception:
        return out
    depth = float(r.x[2])
    noise = float(np.sqrt(np.mean(r.fun ** 2)))
    snr = depth / noise if noise > 0 else 0.0
    shift = float(np.hypot(r.x[0] - cx, r.x[1] - cy))
    out.update(depth=depth, noise=noise, snr=snr)
    if VOID_DEPTH_RANGE[0] <= depth <= VOID_DEPTH_RANGE[1] and snr >= VOID_MIN_SNR and shift <= VOID_MAX_SHIFT_MM:
        out.update(col=x0 + r.x[0] / spacing_mm, row=y0 + r.x[1] / spacing_mm, ok=True, method="fit")
    return out


def find_void_center(arr: np.ndarray, spacing_mm: float, field_center_px: tuple) -> tuple:
    """(col, row) of the void centre - least-squares fit when it can be trusted, otherwise the coarse minimum (see fit_void)."""
    f = fit_void(arr, spacing_mm, field_center_px)
    return f["col"], f["row"]


# ── Field size measurement ────────────────────────────────────────────────────

def _find_field_edge_pair(profile: np.ndarray, spacing_mm: float | None = None) -> tuple:
    """
    Find left and right 50%-penumbra field edges in an inverted 1D portal profile.
    Field = dark (low value), background = bright (high value).
    Returns (left_edge_px, right_edge_px) as sub-pixel floats, or (None, None).
    """
    n = len(profile)
    border = max(5, int(n * 0.08))
    bg_val    = float(np.mean(np.concatenate([profile[:border], profile[-border:]])))
    mid       = n // 2
    mid_hw    = max(5, int(n * 0.10))
    field_val = float(np.mean(profile[max(0, mid - mid_hw) : mid + mid_hw]))
    threshold = (bg_val + field_val) / 2.0

    left_edge = right_edge = None
    for i in range(n - 1):
        if profile[i] >= threshold > profile[i + 1]:
            frac      = (threshold - profile[i]) / (profile[i + 1] - profile[i])
            left_edge = i + frac
            break
    for i in range(n - 1, 0, -1):
        if profile[i] >= threshold > profile[i - 1]:
            frac       = (threshold - profile[i]) / (profile[i - 1] - profile[i])
            right_edge = i - frac
            break
    if spacing_mm and left_edge is not None and right_edge is not None:
        refined = _refine_edge_pair(profile, left_edge, right_edge, spacing_mm)
        if refined is not None:
            return refined
    return left_edge, right_edge


def _refine_edge_pair(profile: np.ndarray, left: float, right: float, spacing_mm: float):
    """
    50 % crossings against the LOCAL levels beside each edge: the level just outside (7-12 mm) and the plateau just inside (4-9 mm), rather
    than one level from the middle of the field.  The middle of these fields is a dip (the phantom void and the beam profile), so the old
    threshold sat too far out and the field read about 2 mm large.  Returns (left, right) or None when a window falls outside the profile.
    """
    m = lambda mm: int(round(mm / spacing_mm))
    n = len(profile)
    li, ri = int(round(left)), int(round(right))
    if li - m(12) < 0 or ri + m(12) >= n or li + m(9) >= n or ri - m(9) < 0 or li - m(6) < 1 or ri + m(6) > n - 2:
        return None
    out_l, in_l = profile[li - m(12):li - m(7)].mean(), profile[li + m(4):li + m(9)].mean()
    out_r, in_r = profile[ri + m(7):ri + m(12)].mean(), profile[ri - m(9):ri - m(4)].mean()
    thr_l, thr_r = (out_l + in_l) / 2.0, (out_r + in_r) / 2.0
    new_l = new_r = None
    for i in range(max(1, li - m(6)), min(n - 1, li + m(6))):
        if profile[i - 1] >= thr_l > profile[i]:
            new_l = i - 1 + (profile[i - 1] - thr_l) / (profile[i - 1] - profile[i])
            break
    for i in range(min(n - 2, ri + m(6)), max(1, ri - m(6)), -1):
        if profile[i + 1] >= thr_r > profile[i]:
            new_r = i + 1 - (profile[i + 1] - thr_r) / (profile[i + 1] - profile[i])
            break
    if new_l is None or new_r is None:
        return None
    return float(new_l), float(new_r)


def measure_field_edges(arr: np.ndarray, spacing_mm: float) -> dict:
    """
    Measure total radiation field size from an Elekta iViewGT portal image using
    50%-penumbra profiles.

    Naming convention (Elekta):
        MLC (Y)  →  horizontal / column direction
                    at G0°/G180°: patient Left/Right (lateral)
                    at G90°/G270°: patient Ant/Post (AP)
        Jaw (X)  →  vertical / row direction = always Superior/Inferior

    Returns the total distance between opposing edges in mm (independent of image
    centre position), plus raw pixel positions for overlay drawing.
    """
    nrows, ncols = arr.shape
    cx_px = (ncols - 1) / 2.0
    cy_px = (nrows - 1) / 2.0

    sigma_px  = max(1.0, 2.0 / spacing_mm)
    h_profile = gaussian_filter1d(arr.mean(axis=0).astype(np.float64), sigma=sigma_px)
    v_profile = gaussian_filter1d(arr.mean(axis=1).astype(np.float64), sigma=sigma_px)

    y_left_px, y_right_px = _find_field_edge_pair(h_profile)   # MLC   (coarse: whole-image averages, only to find the field)
    x_top_px,  x_bot_px   = _find_field_edge_pair(v_profile)   # Jaw

    # Final edges from strips 9-16 mm either side of the field centre (clear of the void, so the dip does not set the threshold), each
    # edge thresholded at 50 % between its own outside level and inside plateau.  Falls back to the coarse edges if a strip fails.
    if None not in (y_left_px, y_right_px, x_top_px, x_bot_px):
        ccx, ccy = (y_left_px + y_right_px) / 2.0, (x_top_px + x_bot_px) / 2.0
        hh = lambda mm: int(round(mm / spacing_mm))
        rows = np.r_[int(ccy) - hh(16):int(ccy) - hh(9), int(ccy) + hh(9):int(ccy) + hh(16)]
        cols = np.r_[int(ccx) - hh(16):int(ccx) - hh(9), int(ccx) + hh(9):int(ccx) + hh(16)]
        if rows.min() >= 0 and rows.max() < nrows and cols.min() >= 0 and cols.max() < ncols:
            a64 = arr.astype(np.float64)
            sg = max(1.0, 1.0 / spacing_mm)
            sh = _find_field_edge_pair(gaussian_filter1d(a64[rows].mean(axis=0), sigma=sg), spacing_mm)
            sv = _find_field_edge_pair(gaussian_filter1d(a64[:, cols].mean(axis=1), sigma=sg), spacing_mm)
            if None not in (*sh, *sv):
                y_left_px, y_right_px = sh
                x_top_px, x_bot_px = sv

    mlc_total_mm = (y_right_px - y_left_px) * spacing_mm \
        if (y_left_px is not None and y_right_px is not None) else None
    jaw_total_mm = (x_bot_px - x_top_px) * spacing_mm \
        if (x_top_px is not None and x_bot_px is not None) else None

    # Penumbra-edge midpoint: most stable field centre for WL analysis.
    # Profile averaging over all rows/columns gives very high SNR and is
    # independent of beam-profile non-uniformity.
    ec_x = (y_left_px + y_right_px) / 2.0 \
        if (y_left_px is not None and y_right_px is not None) else cx_px
    ec_y = (x_top_px + x_bot_px) / 2.0 \
        if (x_top_px is not None and x_bot_px is not None) else cy_px

    return {
        "mlc_total_mm":   mlc_total_mm,
        "jaw_total_mm":   jaw_total_mm,
        "y_left_px":      y_left_px,   "y_right_px": y_right_px,
        "x_top_px":       x_top_px,    "x_bot_px":   x_bot_px,
        "center_px":      (cx_px,      cy_px),       # image centre (legacy)
        "edge_center_px": (ec_x,       ec_y),        # 50% penumbra midpoint
    }


def measure_mlc_leaves(
    arr:           np.ndarray,
    spacing_mm:    float,
    n_leaves:      int   = MLC_LEAVES_PER_BANK,
    leaf_width_mm: float = MLC_LEAF_WIDTH_MM,
) -> list:
    """
    Measure individual MLC leaf tip positions for both banks.

    The MLC leaves are stacked in the X (SI / vertical) direction. The field is
    divided into n_leaves horizontal strips of leaf_width_mm each. For each strip
    the 50%-penumbra left (Y1) and right (Y2) edge positions are found.
    Reference point: geometric image centre.

    Returns list of n_leaves dicts ordered top → bottom (superior → inferior):
        {"leaf_idx": int, "y1_mm": float|None, "y2_mm": float|None}
    """
    nrows, ncols = arr.shape
    cx_px = (ncols - 1) / 2.0
    cy_px = (nrows - 1) / 2.0

    half_field_px = (n_leaves * leaf_width_mm / 2.0) / spacing_mm
    leaf_px       = leaf_width_mm / spacing_mm
    # Keep smoothing to ≤0.5 mm to preserve penumbra sharpness for sub-0.2 mm accuracy.
    # Averaging ~20 rows per strip already suppresses noise; additional Gaussian is minimal.
    sigma_px      = max(0.5, 0.5 / spacing_mm)

    leaves = []
    for i in range(n_leaves):
        r0  = cy_px - half_field_px + i * leaf_px
        r1  = r0 + leaf_px
        r0i = max(0, int(round(r0)))
        r1i = min(nrows, int(round(r1)))
        if r1i <= r0i:
            leaves.append({"leaf_idx": i, "y1_mm": None, "y2_mm": None})
            continue
        strip   = arr[r0i:r1i, :]
        profile = gaussian_filter1d(strip.mean(axis=0).astype(np.float64), sigma=sigma_px)
        y_lp, y_rp = _find_field_edge_pair(profile, spacing_mm)
        leaves.append({
            "leaf_idx":  i,
            "total_mm":  (y_rp - y_lp) * spacing_mm
                         if (y_lp is not None and y_rp is not None) else None,
        })
    return leaves


def analyze_image(ds) -> dict:
    """
    Full analysis of one EPID DICOM image.
    Returns field/void centres (pixels), pixel spacing, and raw displacement (mm).

    Sign convention:
      dx_mm > 0  →  void is to the RIGHT of field centre in the portal image
      dy_mm > 0  →  void is BELOW   field centre in the portal image

    Primary field centre (dx_mm / dy_mm): 50%-penumbra edge midpoint derived from
    column- and row-averaged profiles.  This method averages ~1024 rows/columns,
    giving very high SNR and is independent of beam-profile non-uniformity.
    Secondary references (bbox / centroid) are retained for cross-check.
    """
    arr, spacing = get_pixel_array(ds)
    gantry = float(getattr(ds, "GantryAngle", 0.0))

    # Field edges first — edge_center_px is the primary WL reference centre.
    field_edges = measure_field_edges(arr, spacing)
    edge_fc     = field_edges.get("edge_center_px")   # penumbra midpoint

    # BBox / centroid still computed for cross-check display.
    field_center_bbox_px, field_center_cent_px = find_field_center(arr, spacing)

    # Use edge centre for void search (falls back to bbox if edge detection failed).
    search_fc  = edge_fc if edge_fc is not None else field_center_bbox_px
    void_fit = fit_void(arr, spacing, search_fc)
    void_center_px = (void_fit["col"], void_fit["row"])

    mlc_leaves = measure_mlc_leaves(arr, spacing)

    # Primary displacement: void relative to penumbra edge midpoint.
    if edge_fc is not None:
        dx_px_edge = void_center_px[0] - edge_fc[0]
        dy_px_edge = void_center_px[1] - edge_fc[1]
    else:
        dx_px_edge = void_center_px[0] - field_center_bbox_px[0]
        dy_px_edge = void_center_px[1] - field_center_bbox_px[1]

    dx_px_bbox = void_center_px[0] - field_center_bbox_px[0]
    dy_px_bbox = void_center_px[1] - field_center_bbox_px[1]
    dx_px_cent = void_center_px[0] - field_center_cent_px[0]
    dy_px_cent = void_center_px[1] - field_center_cent_px[1]

    primary_fc = edge_fc if edge_fc is not None else field_center_bbox_px

    _w = None
    for _dt, _tt in (("ContentDate", "ContentTime"), ("StudyDate", "StudyTime")):
        _d, _t = str(getattr(ds, _dt, "") or ""), str(getattr(ds, _tt, "") or "")
        if len(_d) == 8 and _t:
            try:
                _w = datetime.datetime.strptime(_d + _t.split(".")[0].zfill(6)[:6], "%Y%m%d%H%M%S")
                break
            except ValueError:
                pass
    return {
        "sop_uid":              str(getattr(ds, "SOPInstanceUID", "") or ""),
        "when":                 _w,
        "gantry":               gantry,
        "spacing_mm":           spacing,
        "field_center_px":      primary_fc,            # edge midpoint (primary)
        "field_center_bbox_px": field_center_bbox_px,  # BBox (cross-check)
        "field_center_cent_px": field_center_cent_px,  # centroid (cross-check)
        "void_center_px":       void_center_px,
        "void_fit":             {k: void_fit[k] for k in ("depth", "noise", "snr", "ok", "method")},
        "dx_mm":                dx_px_edge * spacing,  # PRIMARY
        "dy_mm":                dy_px_edge * spacing,
        "dx_mm_bbox":           dx_px_bbox * spacing,  # cross-check
        "dy_mm_bbox":           dy_px_bbox * spacing,
        "dx_mm_cent":           dx_px_cent * spacing,
        "dy_mm_cent":           dy_px_cent * spacing,
        "field_edges":          field_edges,
        "mlc_leaves":           mlc_leaves,
        "pixel_array":          arr,
    }


# ── Winston-Lutz metrics ──────────────────────────────────────────────────────

def _minimum_enclosing_circle(points):
    """
    Brute-force minimum enclosing circle (MEC) for ≤6 2D points.

    Returns (cx, cy, radius).  For the 4-cardinal-angle WL test the MEC centre
    is used as the best-fit estimate of the CBCT residual setup error, and the
    MEC radius is the 'walk circle' metric — the smallest circle that contains
    all four void-to-field-centre displacement vectors.
    """
    from itertools import combinations

    def _c2(p, q):
        cx, cy = (p[0]+q[0])/2, (p[1]+q[1])/2
        return cx, cy, float(np.hypot(q[0]-p[0], q[1]-p[1]) / 2)

    def _c3(p, q, s):
        ax, ay = q[0]-p[0], q[1]-p[1]
        bx, by = s[0]-p[0], s[1]-p[1]
        D = 2 * (ax*by - ay*bx)
        if abs(D) < 1e-12:
            return None
        ux = (by*(ax**2+ay**2) - ay*(bx**2+by**2)) / D
        uy = (ax*(bx**2+by**2) - bx*(ax**2+ay**2)) / D
        return p[0]+ux, p[1]+uy, float(np.hypot(ux, uy))

    def _ok(cx, cy, r, pts):
        return all(np.hypot(p[0]-cx, p[1]-cy) <= r + 1e-9 for p in pts)

    pts = list(points)
    best_r, best = float("inf"), (0.0, 0.0, 0.0)

    for i, j in combinations(range(len(pts)), 2):
        cx, cy, r = _c2(pts[i], pts[j])
        if r < best_r and _ok(cx, cy, r, pts):
            best_r, best = r, (cx, cy, r)

    for i, j, k in combinations(range(len(pts)), 3):
        res = _c3(pts[i], pts[j], pts[k])
        if res:
            cx, cy, r = res
            if r < best_r and _ok(cx, cy, r, pts):
                best_r, best = r, (cx, cy, r)

    if np.isinf(best_r):   # degenerate / single point
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        r  = float(max(np.hypot(p[0]-cx, p[1]-cy) for p in pts))
        best = (cx, cy, r)

    return best


# Portal-plane axes versus the patient axes.  The iViewGT panel is FIXED to the gantry, so a static phantom (BB) offset
# b = (X lateral, Y SI, Z AP) shows in the portal image as  u = X cos(G) + Z sin(G),  v = Y :
#   G0: u = +X   G90: u = +Z   G180: u = -X   G270: u = -Z   (v = Y at every angle).
# The offset therefore appears with OPPOSITE signs at opposite gantry angles.  (Checked on 66 stored sessions: the G0 and G180 x-offsets
# are negatively correlated, r = -0.65, and removing a static offset with this model leaves 0.30 mm rms against 0.57 mm for the earlier
# "same sign at every angle" assumption.)
_GANTRY_U = {0: (1.0, 0.0), 90: (0.0, 1.0), 180: (-1.0, 0.0), 270: (0.0, -1.0)}     # (coefficient of X, coefficient of Z)


def _wl_residual_vectors(p, dx: dict, dy: dict):
    """(rel_dx, rel_dy) of each gantry angle after removing the static offset p = (X, Y, Z)."""
    X, Y, Z = p
    return {a: (dx[a] - (cx * X + cz * Z), dy[a] - Y) for a, (cx, cz) in _GANTRY_U.items()}


def _wl_walk(p, dx: dict, dy: dict) -> float:
    return max(float(np.hypot(*v)) for v in _wl_residual_vectors(p, dx, dy).values())


def _fit_minimax_setup(dx: dict, dy: dict) -> tuple[float, float, float]:
    """
    The static phantom offset (X lateral, Y SI, Z AP) that minimises the worst-case residual over the four gantry angles, i.e. the centre
    of the smallest SPHERE touching the four beam axes (pylinac's "gantry 3D isocenter" definition).  Solved exactly: SLSQP on the epigraph
    form (min t subject to every residual <= t) from several starts, the best true worst residual wins; among equal minimisers the smallest
    offset is returned so the answer is reproducible.  Returns (setup_x, setup_y, setup_z).
    """
    from scipy.optimize import minimize as _min

    f = lambda p: _wl_walk(p, dx, dy)
    cons = [{"type": "ineq", "fun": (lambda q, a=a: q[3] - float(np.hypot(*_wl_residual_vectors(q[:3], dx, dy)[a])))} for a in GANTRY_ANGLES]
    bounds = [(-25, 25)] * 3 + [(0, 100)]
    starts = [np.zeros(3), np.array([1.0, 1.0, 1.0]), np.array([-1.0, 1.0, -1.0]), np.array([1.0, -1.0, 1.0]), np.array([-1.0, -1.0, 1.0])]
    starts.append(_min(f, np.zeros(3), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 3000}).x)
    best_x, best = None, np.inf
    for s0 in starts:
        try:
            r = _min(lambda q: q[3], np.r_[s0, f(s0)], method="SLSQP", constraints=cons, bounds=bounds, options={"ftol": 1e-13, "maxiter": 400})
        except Exception:
            continue
        v = f(r.x[:3])
        if np.isfinite(v) and v < best - 1e-12:
            best_x, best = np.asarray(r.x[:3], float), v
    if best_x is None:
        best_x = starts[-1]
        best = f(best_x)
    tmax = best + 1e-6
    try:
        r2 = _min(lambda q: float(np.dot(q[:3], q[:3])), np.r_[best_x, tmax], method="SLSQP", bounds=bounds,
                  constraints=cons + [{"type": "eq", "fun": lambda q: q[3] - tmax}], options={"ftol": 1e-13, "maxiter": 400})
        if f(r2.x[:3]) <= tmax + 1e-5:
            best_x = np.asarray(r2.x[:3], float)
    except Exception:
        pass
    return float(best_x[0]), float(best_x[1]), float(best_x[2])


def compute_wl_results(
    image_results: dict,
    dx_key: str = "dx_mm",
    dy_key: str = "dy_mm",
    use_logs: bool = True,
) -> dict:
    """
    Static phantom set-up offset, per-angle gantry-walk residuals and the isocenter walk (PASS/FAIL).

    Each image gives the void-minus-field-centre displacement (dx, dy) in the portal plane (see _GANTRY_U for how a static phantom offset
    appears at each gantry angle).  The set-up offset (X, Y, Z) is the one that minimises the worst remaining residual; subtracting it from
    each image leaves the mechanical walk residuals.  The walk RADIUS is the largest residual (the radius of the smallest sphere touching the
    four beam axes); that largest residual is the maximum displacement of any beam from the isocenter.  PASS/FAIL uses it against TOLERANCE_MM (1.0 mm).

    dx_key / dy_key select which displacement keys to read from each angle's dict, allowing comparison between field-center detection methods.
    """
    dx = {a: image_results[a][dx_key] for a in GANTRY_ANGLES}
    dy = {a: image_results[a][dy_key] for a in GANTRY_ANGLES}
    # Optional machine log: the aperture the collimator actually had moves the field centre by (actual - set centre of the MLC / jaws); the displacement
    # is void - field centre, so the beam-axis displacement is displacement + that shift. Images without a log are used as measured.
    log_shift, n_logs = {}, 0
    for a in GANTRY_ANGLES:
        ls = (image_results[a].get("log") or {}).get("shift") if use_logs else None
        log_shift[a] = ls
        if ls:
            dx[a] += ls[0]; dy[a] += ls[1]; n_logs += 1

    setup_x, setup_y, setup_z = _fit_minimax_setup(dx, dy)
    res = _wl_residual_vectors((setup_x, setup_y, setup_z), dx, dy)

    per_angle = {}
    for angle in GANTRY_ANGLES:
        per_angle[angle] = {
            "raw_dx": dx[angle],
            "raw_dy": dy[angle],
            "rel_dx": res[angle][0],
            "rel_dy": res[angle][1],
            "log_shift": log_shift[angle],
        }

    walk_r = _wl_walk((setup_x, setup_y, setup_z), dx, dy)

    return {
        "per_angle":      per_angle,
        "setup_x":        setup_x,    # patient lateral (mm)
        "setup_y":        setup_y,    # patient SI (mm)
        "setup_z":        setup_z,    # patient AP (mm)
        "baseline_dx":    setup_x,    # backward-compat alias
        "baseline_dy":    setup_y,
        "baseline_dz":    setup_z,
        "walk_circle_r":  walk_r,
        "walk_diameter_mm": 2.0 * walk_r,      # information only (the diameter of the smallest sphere); not graded
        "logs_used":      n_logs,
        "log_shift_max":  max((float(np.hypot(*v)) for v in log_shift.values() if v), default=0.0),
        "max_2d_walk_mm": walk_r,              # the reported value = the maximum displacement from the isocenter (compared with TOLERANCE_MM)
        "pass_fail":      walk_r <= TOLERANCE_MM,
    }


# ── Diagnostic figure ────────────────────────────────────────────────────────

def generate_diagnostic_figure(img_results: dict, wl_results: dict) -> str | None:
    """
    Render 4 annotated portal images (one per cardinal gantry angle) and save
    as a temporary PNG, returning its path.  Returns None if matplotlib is
    unavailable.

    Each panel shows:
      • Gray-scale portal image windowed to the field interior
      • Cyan dashed box  — radiation field boundary
      • Cyan dotted crosshair — field centre
      • Red +  — detected void centre
      • Red circle — expected 6.4 mm void diameter
      • Yellow arrow — displacement vector (field → void)
      • White scale bar (5 mm at isocenter)
    """
    if not _MPL_OK:
        return None

    import tempfile

    walk_r = wl_results["walk_circle_r"]
    passed = wl_results["pass_fail"]

    # 4 panels: annotated portal images only
    fig = plt.figure(figsize=(22.4, 6.4), facecolor="#111111", layout="constrained")
    gs  = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.07)
    portal_axes = [fig.add_subplot(gs[i]) for i in range(4)]

    fig.suptitle(
        f"Portal Images — Field & Void Centre Detection     "
        f"Max displacement = {walk_r:.3f} mm   "
        f"{'PASS ✓' if passed else 'FAIL ✗'}",
        color="white", fontsize=11, fontweight="bold",
    )

    angle_colors = {0: "#64b5f6", 90: "#81c784", 180: "#ffb74d", 270: "#f06292"}

    for ax, angle in zip(portal_axes, GANTRY_ANGLES):
        r   = img_results[angle]
        arr = r["pixel_array"]
        fc  = r["field_center_px"]   # (col, row)
        vc  = r["void_center_px"]
        spc = r["spacing_mm"]        # isocenter mm/px
        wla = wl_results["per_angle"][angle]
        corr_dr = np.sqrt(wla["rel_dx"] ** 2 + wla["rel_dy"] ** 2)

        # Tight crop: field half-width + 20 % margin
        field_half_px = (FIELD_SIZE_MM / 2.0) / spc
        margin = max(15, int(field_half_px * 0.20))
        half   = int(field_half_px) + margin
        fcx, fcy = int(round(fc[0])), int(round(fc[1]))
        c0 = max(0,            fcx - half)
        c1 = min(arr.shape[1], fcx + half)
        r0 = max(0,            fcy - half)
        r1 = min(arr.shape[0], fcy + half)
        crop = arr[r0:r1, c0:c1]

        # Window tightly around the air void itself so it stands out clearly.
        # Elekta inverted: field = LOW value, background = HIGH value; the void
        # is a subtle local minimum riding on top of the much larger field→
        # background range, so windowing off the whole crop (as before) buries
        # it in mid-gray. Instead stretch contrast using only a small ROI
        # centred on the detected void — this saturates the field edge/
        # background to black/white but maximises void visibility.
        void_r_px_est = (VOID_DIAMETER_MM / 2.0) / spc
        win_half = max(20, int(void_r_px_est * 5))
        vcx, vcy = int(round(vc[0])), int(round(vc[1]))
        wc0 = max(0,            vcx - win_half)
        wc1 = min(arr.shape[1], vcx + win_half)
        wr0 = max(0,            vcy - win_half)
        wr1 = min(arr.shape[0], vcy + win_half)
        void_roi = arr[wr0:wr1, wc0:wc1].astype(np.float64)
        vmin_c = float(np.percentile(void_roi, 1))
        vmax_c = float(np.percentile(void_roi, 97))

        ax.imshow(
            crop, cmap="gray", vmin=vmin_c, vmax=vmax_c,
            aspect="equal", extent=[c0, c1, r1, r0],
            interpolation="bilinear",
        )

        # 40 × 40 mm field boundary box
        fh = field_half_px
        ax.add_patch(mpatches.Rectangle(
            (fc[0] - fh, fc[1] - fh), 2 * fh, 2 * fh,
            edgecolor="cyan", facecolor="none", lw=1.2, ls="--", alpha=0.8,
        ))

        # Field centre crosshairs — penumbra edge midpoint (cyan) and geometric centroid (orange)
        ax.axhline(fc[1], color="cyan",   lw=0.8, ls=":",  alpha=0.7)
        ax.axvline(fc[0], color="cyan",   lw=0.8, ls=":",  alpha=0.7)
        fcc = r.get("field_center_cent_px")
        if fcc is not None:
            ax.axhline(fcc[1], color="orange", lw=0.8, ls="-.", alpha=0.6)
            ax.axvline(fcc[0], color="orange", lw=0.8, ls="-.", alpha=0.6)
            dcx = (fcc[0] - fc[0]) * spc
            dcy = (fcc[1] - fc[1]) * spc
            ax.text(
                0.02, 0.02,
                f"ΔFc (cent−edge)  x={dcx:+.3f}  y={dcy:+.3f} mm",
                transform=ax.transAxes,
                color="white", fontsize=6.0, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0),
            )

        # Air void circle (6.4 mm diameter) centred on detected void
        void_r_px = (VOID_DIAMETER_MM / 2.0) / spc
        col = angle_colors[angle]
        ax.plot(vc[0], vc[1], "+", color=col, ms=16, mew=2.2, zorder=5)
        ax.add_patch(mpatches.Circle(
            (vc[0], vc[1]), void_r_px,
            edgecolor=col, facecolor="none", lw=2.0, zorder=4,
        ))

        # Displacement arrow (field centre → void centre)
        ax.annotate(
            "", xy=(vc[0], vc[1]), xytext=(fc[0], fc[1]),
            arrowprops=dict(arrowstyle="->", color="#ffff00", lw=1.8),
        )

        # ── Detected field edges overlay (lime green) ─────────────────────────
        fe = r.get("field_edges", {})
        if fe:
            cx_f, cy_f = fe.get("center_px", (0, 0))
            y_lp = fe.get("y_left_px");  y_rp = fe.get("y_right_px")
            x_tp = fe.get("x_top_px");   x_bp = fe.get("x_bot_px")
            # Image-centre marker (white ×)
            if c0 <= cx_f <= c1 and r0 <= cy_f <= r1:
                ax.plot(cx_f, cy_f, "wx", ms=9, mew=1.5, zorder=6, alpha=0.9)
            # MLC edges (Y1 left / Y2 right) — lime vertical lines
            for xpx in [y_lp, y_rp]:
                if xpx is not None and c0 <= xpx <= c1:
                    ax.axvline(xpx, color="#00e676", lw=1.3, ls="-", alpha=0.80)
            # Jaw edges (X1 sup / X2 inf) — lime horizontal lines
            for ypx in [x_tp, x_bp]:
                if ypx is not None and r0 <= ypx <= r1:
                    ax.axhline(ypx, color="#00e676", lw=1.3, ls="-", alpha=0.80)
            # Field-size annotation (bottom-right corner)
            mlc_t = fe.get("mlc_total_mm")
            jaw_t = fe.get("jaw_total_mm")
            if mlc_t is not None or jaw_t is not None:
                mlc_dir = "AP" if angle in (90, 270) else "Lat"
                lines = []
                if mlc_t is not None:
                    lines.append(f"MLC ({mlc_dir}): {mlc_t:.2f} mm")
                if jaw_t is not None:
                    lines.append(f"Jaw (SI):    {jaw_t:.2f} mm")
                ax.text(
                    0.98, 0.02, "\n".join(lines),
                    transform=ax.transAxes,
                    color="#00e676", fontsize=6.0, va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0),
                )

        # 5 mm scale bar
        bar_px = 5.0 / spc
        bx0    = c0 + 8
        by     = r1 - 10
        ax.plot([bx0, bx0 + bar_px], [by, by], color="white", lw=2.0)
        ax.text(bx0 + bar_px / 2, by - 4, "5 mm",
                color="white", ha="center", va="bottom", fontsize=6.5)

        # Panel title (colour = angle colour, status = corrected walk magnitude)
        corr_label = f"{corr_dr:.3f} mm"
        x_dir = "AP" if angle in (90, 270) else "Lat"
        ax.set_title(
            f"G{angle:03d}°  |  Raw |ΔR|={np.hypot(wla['raw_dx'],wla['raw_dy']):.3f} mm\n"
            f"Raw  Δ{x_dir}={wla['raw_dx']:+.3f}  ΔSI={wla['raw_dy']:+.3f} mm\n"
            f"Corr Δ{x_dir}={wla['rel_dx']:+.3f}  ΔSI={wla['rel_dy']:+.3f}  [{corr_label}]",
            color=col, fontsize=7.5, fontweight="bold",
        )
        ax.tick_params(colors="#777777", labelsize=6)
        for sp in ax.spines.values():
            sp.set_color("#333333")
        ax.set_facecolor("#111111")

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return tmp.name


def generate_walk_circle_figure(wl_results: dict) -> str | None:
    """
    Render the 2-D isocenter walk displacement map as a standalone PNG.
    Shows corrected per-angle vectors, MEC walk circle, tolerance ring, and legend.
    Returns the temp-file path, or None if matplotlib is unavailable.
    """
    if not _MPL_OK:
        return None

    import tempfile
    import matplotlib.patches as mpatches

    per_angle = wl_results["per_angle"]
    walk_r    = wl_results["walk_circle_r"]
    passed    = wl_results["pass_fail"]

    mec_cx, mec_cy = 0.0, 0.0          # the residuals are measured from the fitted isocenter (the circle centre)

    angle_colors = {0: "#64b5f6", 90: "#81c784", 180: "#ffb74d", 270: "#f06292"}
    angle_labels = {0: "G000° Lat", 90: "G090° AP", 180: "G180° Lat", 270: "G270° AP"}

    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor="#111111")
    ax.set_facecolor("#1a1a1a")

    # Grid
    ax.grid(True, color="#2a2a2a", lw=0.6, zorder=0)

    # Isocenter crosshair
    ax.axhline(0, color="#444444", lw=0.8, ls=":", zorder=1)
    ax.axvline(0, color="#444444", lw=0.8, ls=":", zorder=1)

    # Tolerance circle (dashed red)
    ax.add_patch(mpatches.Circle(
        (0, 0), TOLERANCE_MM, color="#ef5350", fill=False,
        ls="--", lw=1.5, alpha=0.85, zorder=2,
    ))
    ax.text(
        TOLERANCE_MM * 0.707 + 0.03, TOLERANCE_MM * 0.707 + 0.03,
        f"Limit {TOLERANCE_MM:.1f} mm", color="#ef5350", fontsize=7.5, zorder=3,
    )

    # MEC walk circle (white)
    ax.add_patch(mpatches.Circle(
        (mec_cx, mec_cy), walk_r, color="white", fill=False,
        lw=2.0, alpha=0.9, zorder=4,
    ))

    # MEC centre cross
    ax.plot(mec_cx, mec_cy, "+", color="#90caf9", ms=14, mew=2.5, zorder=6,
            label="fitted isocenter")

    # Spokes from MEC centre to each point
    for angle in GANTRY_ANGLES:
        pt  = per_angle[angle]
        col = angle_colors[angle]
        ax.plot([mec_cx, pt["rel_dx"]], [mec_cy, pt["rel_dy"]],
                color=col, lw=0.9, alpha=0.45, zorder=3)

    # Per-angle dots
    for angle in GANTRY_ANGLES:
        pt  = per_angle[angle]
        col = angle_colors[angle]
        ax.plot(
            pt["rel_dx"], pt["rel_dy"], "o",
            color=col, ms=11, zorder=7, markeredgecolor="white", markeredgewidth=0.6,
            label=f"{angle_labels[angle]}  ({pt['rel_dx']:+.3f}, {pt['rel_dy']:+.3f}) mm",
        )

    # Auto-scale with margin
    all_vals = [per_angle[a]["rel_dx"] for a in GANTRY_ANGLES] + \
               [per_angle[a]["rel_dy"] for a in GANTRY_ANGLES] + [mec_cx, mec_cy]
    lim = max(abs(v) for v in all_vals) + max(0.25, walk_r * 0.4)
    lim = max(lim, TOLERANCE_MM * 1.25)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    pf_color = "#66bb6a" if passed else "#ef5350"
    ax.set_title(
        f"Max displacement from isocenter   {walk_r:.3f} mm   "
        f"{'PASS ✓' if passed else 'FAIL ✗'}",
        color=pf_color, fontsize=10, fontweight="bold", pad=6,
    )
    ax.set_xlabel("ΔLat / ΔAP  (mm)", color="#bbbbbb", fontsize=9)
    ax.set_ylabel("ΔSI  (mm)",         color="#bbbbbb", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Walk circle radius annotation
    ax.annotate(
        f"r = {walk_r:.3f} mm  (Ø {2 * walk_r:.3f} mm)",
        xy=(mec_cx + walk_r * 0.707, mec_cy + walk_r * 0.707),
        color="white", fontsize=8, zorder=8,
        xytext=(4, 4), textcoords="offset points",
    )

    legend = ax.legend(
        loc="lower right", fontsize=7.5,
        facecolor="#1f1f1f", labelcolor="white",
        edgecolor="#555555", framealpha=0.92,
        handlelength=1.2, handleheight=1.0,
    )

    fig.tight_layout(pad=0.6)

    tmp = tempfile.NamedTemporaryFile(suffix="_walk_circle.png", delete=False)
    plt.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return tmp.name


# ── Picket Fence ─────────────────────────────────────────────────────────────

def identify_pf_dicom(ds) -> bool:
    """Return True if this DICOM is a picket fence portal image."""
    label = str(getattr(ds, "RTImageLabel",       "") or "").upper()
    desc  = str(getattr(ds, "SeriesDescription",  "") or "").upper()
    return "PF" in label or "PICKET" in label or "PF" in desc.split()


def _find_pf_edge(profile: np.ndarray, threshold: float, rising: bool):
    """Sub-pixel edge position where profile crosses threshold."""
    if rising:
        for i in range(1, len(profile)):
            if profile[i] >= threshold > profile[i - 1]:
                span = profile[i] - profile[i - 1]
                return i - 1 + (threshold - profile[i - 1]) / span if span else float(i)
    else:
        for i in range(len(profile) - 2, -1, -1):
            if profile[i] >= threshold > profile[i + 1]:
                span = profile[i] - profile[i + 1]
                return i + 1 - (threshold - profile[i + 1]) / span if span else float(i)
    return None


def analyze_picket_fence(ds) -> dict:
    """
    Analyse a single-picket EPID portal image (1 cm gap, 20 cm length,
    central 80 leaves = 40 per bank at 5 mm each).

    Returns a dict with per-leaf deviations and summary statistics.
    """
    from scipy.ndimage import gaussian_filter

    arr, spacing_iso = get_pixel_array(ds)   # arr in raw counts; spacing in mm/px at iso
    rows, cols = arr.shape

    # Invert: low pixel = high dose → normalised so high dose = 1.0
    arr_f  = arr.astype(np.float32)
    inv    = (arr_f.max() - arr_f) / max(arr_f.max() - arr_f.min(), 1.0)
    smooth = gaussian_filter(inv, sigma=1.5)

    # Detect field bounding box.
    # The picket is narrow (≈1 cm / ~40 px out of 1024) so the row-direction
    # profile must be taken over the column band only, not the full image width.
    threshold_detect = 0.12
    col_prof = smooth.mean(axis=0)          # average along rows → column profile
    c_mask   = col_prof > threshold_detect
    if not c_mask.any():
        raise ValueError("Picket Fence: cannot detect field columns in DICOM image.")
    c_where = np.where(c_mask)[0]
    c0, c1  = int(c_where[0]), int(c_where[-1])

    # Row profile: take the max within the detected column band so the narrow
    # picket isn't diluted by background pixels in the rest of the image.
    row_prof = smooth[:, c0:c1 + 1].max(axis=1)
    r_mask   = row_prof > threshold_detect
    if not r_mask.any():
        raise ValueError("Picket Fence: cannot detect field rows in DICOM image.")
    r_where = np.where(r_mask)[0]
    r0, r1  = int(r_where[0]), int(r_where[-1])

    field_px_rows = r1 - r0
    leaf_px       = PF_LEAF_WIDTH_MM / spacing_iso        # px per leaf
    # Cap at the expected leaf pair count: the 12% detection threshold sits
    # outside the geometric field edge so the measured height is slightly
    # taller than nominal, causing round() to give one extra leaf.
    n_leaves      = min(max(1, round(field_px_rows / leaf_px)), PF_LEAVES_TOTAL // 2)

    # Expanded column window for edge detection (include penumbra on each side)
    pad_col = max(8, int(8.0 / spacing_iso))
    col_lo  = max(0, c0 - pad_col)
    col_hi  = min(cols - 1, c1 + pad_col)

    centers, left_edges, right_edges, leaf_y_mm = [], [], [], []
    field_centre_row = (r0 + r1) / 2.0

    for i in range(n_leaves):
        rl = r0 + int(round(i       * field_px_rows / n_leaves))
        rr = r0 + int(round((i + 1) * field_px_rows / n_leaves))
        rr = min(rr, rows)
        if rr <= rl:
            continue

        profile = smooth[rl:rr, col_lo:col_hi].mean(axis=0)
        peak = float(profile.max())
        if peak < 0.08:
            continue

        half = peak * 0.50
        le   = _find_pf_edge(profile, half, rising=True)
        re   = _find_pf_edge(profile, half, rising=False)
        if le is None or re is None:
            continue

        centers.append(col_lo + (le + re) / 2.0)
        left_edges.append(col_lo + le)
        right_edges.append(col_lo + re)
        leaf_y_mm.append(((rl + rr) / 2.0 - field_centre_row) * spacing_iso)

    if not centers:
        raise ValueError("Picket Fence: no leaf edges could be detected.")

    centers      = np.array(centers,      dtype=np.float64)
    left_edges   = np.array(left_edges,   dtype=np.float64)
    right_edges  = np.array(right_edges,  dtype=np.float64)
    leaf_y_mm    = np.array(leaf_y_mm,    dtype=np.float64)

    nominal_col      = float(centers.mean())
    deviations_mm    = (centers - nominal_col) * spacing_iso
    widths_mm        = (right_edges - left_edges) * spacing_iso
    max_dev          = float(np.abs(deviations_mm).max())

    # Study date/time from DICOM
    pf_date, pf_time = None, None
    for dtag, ttag in (("StudyDate", "StudyTime"), ("ContentDate", "ContentTime")):
        raw_d = getattr(ds, dtag, None)
        if raw_d and len(str(raw_d)) == 8:
            try:
                pf_date = datetime.datetime.strptime(str(raw_d), "%Y%m%d").date()
                raw_t   = getattr(ds, ttag, None)
                if raw_t:
                    ts = str(raw_t).split(".")[0].zfill(6)
                    if len(ts) >= 6:
                        pf_time = datetime.time(int(ts[0:2]), int(ts[2:4]), int(ts[4:6]))
            except (ValueError, TypeError):
                pass
            if pf_date:
                break

    return {
        "leaf_y_mm":          leaf_y_mm,
        "leaf_deviations_mm": deviations_mm,
        "leaf_widths_mm":     widths_mm,
        "nominal_col":        nominal_col,
        "max_deviation_mm":   max_dev,
        "rms_deviation_mm":   float(np.sqrt(np.mean(deviations_mm ** 2))),
        "n_leaves":           len(centers),
        "pass_fail":          max_dev <= PF_TOLERANCE_MM,
        "spacing_iso_mm":     float(spacing_iso),
        "field_bounds":       (r0, r1, c0, c1),
        "_image_array":       arr_f,
        "_inv_array":         inv,    # normalised 0-1, high dose = high value
        "study_date":         pf_date,
        "study_time":         pf_time,
    }


def pf_avg_width_central_mm(pf_results: dict,
                             half_range_mm: float = PF_AVG_WIDTH_HALF_RANGE_MM) -> float:
    """
    Mean measured leaf width over leaves within +/- half_range_mm of field
    centre only. Leaves near the ends of the 200 mm picket sit closer to the
    longitudinal field edge, where MV penumbra widens the 50%-threshold
    measurement — excluding them keeps avg_width representative of the
    central, unconfounded picket body.
    """
    widths = pf_results["leaf_widths_mm"]
    y      = pf_results["leaf_y_mm"]
    if len(widths) == 0:
        return 0.0
    mask = np.abs(y) <= half_range_mm
    if not mask.any():
        mask = np.ones_like(widths, dtype=bool)
    return float(np.mean(widths[mask]))


def generate_pf_figure(pf_results: dict) -> str | None:
    """Two-panel diagnostic figure: portal image (left) + leaf deviation chart (right)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile

        inv         = pf_results["_inv_array"]      # normalised 0-1, high dose = bright
        r0, r1, c0, c1 = pf_results["field_bounds"]
        sp          = pf_results["spacing_iso_mm"]
        n_leaves    = pf_results["n_leaves"]
        deviations  = pf_results["leaf_deviations_mm"]
        leaf_y      = pf_results["leaf_y_mm"]
        nominal_col = pf_results["nominal_col"]
        max_dev     = pf_results["max_deviation_mm"]
        rms_dev     = pf_results["rms_deviation_mm"]
        passed      = pf_results["pass_fail"]

        fig, (ax_img, ax_dev) = plt.subplots(
            1, 2, figsize=(13, 9),
            gridspec_kw={"width_ratios": [1, 2.5]},
        )
        fig.patch.set_facecolor("#1e1e1e")

        # ── Portal image panel ────────────────────────────────────────────────
        pad_r = max(10, int(15 / sp))
        pad_c = max(20, int(30 / sp))
        row_lo = max(0, r0 - pad_r);  row_hi = min(inv.shape[0], r1 + pad_r)
        col_lo = max(0, c0 - pad_c);  col_hi = min(inv.shape[1], c1 + pad_c)

        # Use inverted (dose = bright) array so background is dark, no blowout.
        # Window from the central 90% of field rows to exclude end-of-picket
        # penumbra / hot-spot artefacts that would otherwise clip the display.
        crop_inv  = inv[row_lo:row_hi, col_lo:col_hi]
        trim_r    = max(1, (r1 - r0) // 20)
        core_inv  = inv[r0 + trim_r : r1 - trim_r + 1, c0 : c1 + 1]
        vlo = float(np.percentile(core_inv, 5))
        vhi = float(np.percentile(core_inv, 95))

        ax_img.imshow(crop_inv, cmap="gray", vmin=vlo, vmax=vhi,
                      aspect="equal", origin="upper")
        ax_img.set_facecolor("#1e1e1e")
        ax_img.set_axis_off()

        field_rows   = r1 - r0
        nom_crop_col = nominal_col - col_lo

        # Leaf grid lines
        for i in range(n_leaves + 1):
            row_crop = r0 + i * (field_rows / n_leaves) - row_lo
            ax_img.axhline(row_crop, color="#80cbc4", alpha=0.30,
                           linewidth=0.5, zorder=2)

        # Nominal centre line
        ax_img.axvline(nom_crop_col, color="yellow", linewidth=1.0,
                       alpha=0.65, linestyle="--", zorder=3)

        # Per-leaf deviation dots — green / orange (warn) / red (fail)
        for i, (_, dev) in enumerate(zip(leaf_y, deviations)):
            row_crop = r0 + i * (field_rows / n_leaves) + \
                       (field_rows / n_leaves) / 2 - row_lo
            col_crop = nom_crop_col + dev / sp
            c = "#4caf50" if abs(dev) < PF_WARN_MM else \
                "#ff9800" if abs(dev) < PF_TOLERANCE_MM else "#f44336"
            ax_img.plot(col_crop, row_crop, "o", color=c,
                        markersize=3.5, markeredgecolor="#111111",
                        markeredgewidth=0.4, zorder=5)

        ax_img.set_title("Portal Image", color="white", fontsize=11, pad=4)

        # ── Deviation chart ───────────────────────────────────────────────────
        ax_dev.set_facecolor("#252525")
        for sp_edge in ax_dev.spines.values():
            sp_edge.set_color("#555555")

        for y, dev in zip(leaf_y, deviations):
            c = "#4caf50" if abs(dev) < PF_WARN_MM else \
                "#ff9800" if abs(dev) < PF_TOLERANCE_MM else "#f44336"
            ax_dev.barh(y, dev, height=PF_LEAF_WIDTH_MM * 0.80,
                        color=c, edgecolor="#1e1e1e", linewidth=0.4, zorder=3)

        ax_dev.axvline(0, color="white", linewidth=1.0, alpha=0.55, zorder=4)
        # Warning lines (orange)
        ax_dev.axvline(+PF_WARN_MM, color="#ff9800", linewidth=1.2,
                       linestyle="--", zorder=4,
                       label=f"±{PF_WARN_MM:.1f} mm warning")
        ax_dev.axvline(-PF_WARN_MM, color="#ff9800", linewidth=1.2,
                       linestyle="--", zorder=4)
        # Fail lines (red)
        ax_dev.axvline(+PF_TOLERANCE_MM, color="#ef5350", linewidth=1.4,
                       linestyle="--", zorder=4,
                       label=f"±{PF_TOLERANCE_MM:.1f} mm fail")
        ax_dev.axvline(-PF_TOLERANCE_MM, color="#ef5350", linewidth=1.4,
                       linestyle="--", zorder=4)

        ax_dev.set_xlabel("Deviation from Nominal Centre (mm)",
                          color="white", fontsize=10)
        ax_dev.set_ylabel("Leaf Position  —  SI direction (mm)",
                          color="white", fontsize=10)
        ax_dev.set_title("Per-Leaf Centre Deviation",
                         color="white", fontsize=11, pad=4)
        ax_dev.tick_params(colors="white", labelsize=9)
        ax_dev.invert_yaxis()   # superior (negative Y) at top

        result_col = "#4caf50" if passed else "#ef5350"
        ax_dev.text(
            0.98, 0.02,
            f"Max |Δ| = {max_dev:.2f} mm\n"
            f"RMS   = {rms_dev:.2f} mm\n"
            f"Leaves = {n_leaves}\n"
            f"{'PASS ✓' if passed else 'FAIL ✗'}",
            transform=ax_dev.transAxes, ha="right", va="bottom",
            color=result_col, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e1e",
                      edgecolor="#555555", alpha=0.85),
        )
        ax_dev.legend(loc="upper right", framealpha=0.3,
                      labelcolor="white", fontsize=9,
                      facecolor="#1e1e1e", edgecolor="#555555")

        # Zoom in: show ±(2× fail tol) or data max with margin, min 0.3 mm
        xlim = max(PF_TOLERANCE_MM * 2.0, max_dev * 1.5, 0.3)
        ax_dev.set_xlim(-xlim, xlim)

        pf_title_col = "#66bb6a" if passed else "#ef5350"
        fig.suptitle(
            f"Picket Fence — MLC Leaf Position Accuracy   "
            f"{'PASS ✓' if passed else 'FAIL ✗'}",
            color=pf_title_col, fontsize=13, fontweight="bold", y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        tmp = tempfile.NamedTemporaryFile(suffix="_pf.png", delete=False)
        fig.savefig(tmp.name, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return tmp.name
    except Exception as exc:
        print(f"PF figure error: {exc}")
        return None


# ── GUI ───────────────────────────────────────────────────────────────────────

def _apply_dark_theme(app: QApplication) -> None:
    """Apply a dark Fusion-based palette and stylesheet to the application."""
    from PySide6.QtGui import QPalette
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(43,  43,  43))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Base,            QColor(30,  30,  30))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(43,  43,  43))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(43,  43,  43))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Button,          QColor(60,  60,  60))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.BrightText,      Qt.GlobalColor.white)
    pal.setColor(QPalette.ColorRole.Link,            QColor(30,  120, 200))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(31,  83,  141))
    pal.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    app.setPalette(pal)
    app.setStyleSheet("""
        QPushButton {
            background-color: #1f538d; color: white; border-radius: 6px;
            padding: 6px 14px; font-size: 14px; min-height: 28px;
        }
        QPushButton:hover   { background-color: #2563ae; }
        QPushButton:pressed { background-color: #174078; }
        QPushButton:disabled { background-color: #444444; color: #777777; }
        QComboBox {
            background-color: #3a3a3a; color: #dcdcdc;
            border: 1px solid #555555; border-radius: 4px;
            padding: 4px 8px; font-size: 14px; min-height: 28px;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background-color: #3a3a3a; color: #dcdcdc;
            selection-background-color: #1f538d;
        }
        QTabWidget::pane { border: 1px solid #3a3a3a; border-radius: 4px; }
        QTabBar::tab {
            background-color: #3a3a3a; color: #aaaaaa;
            padding: 8px 20px; font-size: 13px;
        }
        QTabBar::tab:selected          { background-color: #1f538d; color: white; }
        QTabBar::tab:hover:!selected   { background-color: #4a4a4a; }
        QScrollArea { border: none; }
        QTableWidget {
            background-color: #2b2b2b; color: #dcdcdc;
            gridline-color: #3a3a3a; border: none;
        }
        QTableWidget::item { padding: 4px; }
        QHeaderView::section {
            background-color: #3a3a3a; color: #aaaaaa;
            padding: 6px; border: none; font-size: 13px; font-weight: bold;
        }
        QScrollBar:vertical   { background: #2b2b2b; width: 12px; }
        QScrollBar:horizontal { background: #2b2b2b; height: 12px; }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: #555555; border-radius: 6px; min-length: 20px;
        }
        QDialog { background-color: #2b2b2b; }
    """)


class EpidSessionsDialog(QDialog):
    """New Daily WL sessions found in the EPID folder, grouped by day. Review each session in the main window (images, results, report), one after another, or import them all."""

    def __init__(self, parent, sessions, machine):
        super().__init__(parent)
        self._app, self._machine = parent, machine
        self._all = sessions
        self.setWindowTitle(f"EPID folder — {machine}")
        self.resize(980, 560)
        lay = QVBoxLayout(self)
        self._summary = QLabel("")
        self._summary.setStyleSheet("font-size: 14px; font-weight: bold;")
        lay.addWidget(self._summary)
        lay.addWidget(QLabel("Sessions are grouped by day. Select one and open it to review its images and results and to save its report; "
                             "\"Review next\" in the main window then steps through the rest."))
        self._show_done = QCheckBox("Show sessions already processed or set aside")
        self._show_done.toggled.connect(self._fill)
        lay.addWidget(self._show_done)
        self._tree = QTreeWidget()
        self._tree.setColumnCount(8)
        self._tree.setHeaderLabels(["Day / session", "Images", "Max displacement", "MLC size (Δ ref)", "Jaw size (Δ ref)", "Picket fence max", "Machine log", "Status"])
        self._tree.header().setSectionResizeMode(QHeaderView.Stretch)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._tree.setSelectionMode(QTreeWidget.SingleSelection)
        self._tree.setRootIsDecorated(True)
        self._tree.itemDoubleClicked.connect(lambda *_: self._open())
        lay.addWidget(self._tree, 1)
        row = QHBoxLayout()
        b_all = QPushButton("Import all new sessions…"); b_all.clicked.connect(self._import_all)
        b_all.setToolTip("Analyse and save every new, complete session in the list (no report PDFs): the Winston-Lutz record, field sizes and the picket fence go into the trend history. "
                         "Can be stopped and resumed; sessions that cannot be analysed reliably are listed and left here for review.")
        b_open = QPushButton("Open selected"); b_open.clicked.connect(self._open)
        b_open.setToolTip("Analyse the selected session (or the first of the selected day) and show it in the main window.")
        b_rep = QPushButton("Open and save report…"); b_rep.clicked.connect(lambda: self._open(report=True))
        b_rep.setToolTip("Open the selected session and go straight to saving its report (which also records it in the history).")
        b_skip = QPushButton("Set aside"); b_skip.setToolTip("Do not offer this session (or every session of the selected day) again"); b_skip.clicked.connect(self._skip)
        b_close = QPushButton("Close"); b_close.clicked.connect(self.reject)
        for wdg in (b_all, b_open, b_rep, b_skip):
            row.addWidget(wdg)
        row.addStretch(); row.addWidget(b_close)
        lay.addLayout(row)
        self._fill()

    def _rows(self):
        return [x for x in self._all if self._show_done.isChecked() or x.status == "new"]

    def _days(self):
        out = {}
        for x in sorted(self._rows(), key=lambda x: x.when):
            out.setdefault(x.when.date(), []).append(x)
        return out

    def _fill(self, *_):
        self._tree.clear()
        refs = self._app._get_field_refs(self._machine)
        lvl = lambda dev, warn, fail: "#66bb6a" if abs(dev) <= warn else "#ffa726" if abs(dev) <= fail else "#ef5350"
        days = self._days()
        n_new = sum(1 for x in self._all if x.status == "new")
        self._summary.setText(f"{n_new} new session(s) on {len([d for d, v in days.items() if any(x.status == 'new' for x in v)])} day(s) — {self._machine}")
        bold = QFont(); bold.setBold(True)
        for day, xs in days.items():
            prev = [x.preview for x in xs if x.preview]
            worst = max((p["displacement"] for p in prev), default=None)
            head = QTreeWidgetItem([f"{day:%a %Y-%m-%d}  ({len(xs)})", "", "—" if worst is None else f"worst {worst:.2f} mm", "", "", "", "", ""])
            for c in range(8):
                head.setFont(c, bold)
            if worst is not None:
                head.setForeground(2, QBrush(QColor("#66bb6a" if worst <= TOLERANCE_MM else "#ef5350")))
            head.setData(0, Qt.ItemDataRole.UserRole, None)
            self._tree.addTopLevelItem(head)
            for x in xs:
                imgs = "4 of 4" if x.complete else f"{4 - len(x.missing)} of 4 (missing G{', G'.join(str(a) for a in x.missing)})"
                colors_ = {}
                if x.preview:
                    p_ = x.preview
                    res = f"{p_['displacement']:.2f} mm  {'PASS' if p_['pass'] else 'FAIL'}"
                    logs = f"{p_['logs']} of 4 images"
                    mlc = "—" if p_.get("mlc") is None else f"{p_['mlc']:.2f} ({p_['mlc'] - refs['mlc']:+.2f})"
                    jaw = "—" if p_.get("jaw") is None else f"{p_['jaw']:.2f} ({p_['jaw'] - refs['jaw']:+.2f})"
                    pfm = "—" if p_.get("pf_max") is None else f"{p_['pf_max']:.2f} mm"
                    colors_ = {2: "#66bb6a" if p_["pass"] else "#ef5350"}
                    if p_.get("mlc") is not None:
                        colors_[3] = lvl(p_["mlc"] - refs["mlc"], FIELD_SIZE_WARN_MM, FIELD_SIZE_FAIL_MM)
                    if p_.get("jaw") is not None:
                        colors_[4] = lvl(p_["jaw"] - refs["jaw"], FIELD_SIZE_JAW_WARN_MM, FIELD_SIZE_JAW_FAIL_MM)
                    if p_.get("pf_max") is not None:
                        colors_[5] = "#66bb6a" if p_["pf_max"] < PF_WARN_MM else "#ffa726" if p_["pf_max"] < PF_TOLERANCE_MM else "#ef5350"
                else:
                    res = logs = mlc = jaw = pfm = "—"
                it = QTreeWidgetItem([f"{x.when:%H:%M}", imgs, res, mlc, jaw, pfm, logs, x.status])
                it.setData(0, Qt.ItemDataRole.UserRole, x)
                for c, col in colors_.items():
                    it.setForeground(c, QBrush(QColor(col)))
                head.addChild(it)
            head.setExpanded(True)
        first = next((self._tree.topLevelItem(i).child(0) for i in range(self._tree.topLevelItemCount()) if self._tree.topLevelItem(i).childCount()), None)
        if first is not None:
            self._tree.setCurrentItem(first)

    def _sessions_of(self, item):
        """The session of a session row, or all the sessions of a day row."""
        if item is None:
            return []
        x = item.data(0, Qt.ItemDataRole.UserRole)
        if x is not None:
            return [x]
        return [item.child(i).data(0, Qt.ItemDataRole.UserRole) for i in range(item.childCount())]

    def _current(self):
        xs = self._sessions_of(self._tree.currentItem())
        return xs[0] if xs else None

    def _open(self, report=False):
        x = self._current()
        if x is None:
            return
        if not x.complete:
            QMessageBox.information(self, "Incomplete session", f"This session has no image at G{', G'.join(str(a) for a in x.missing)}.")
            return
        queue = sorted((y for y in self._all if y.status == "new" and y.complete), key=lambda y: y.when)
        self.accept()
        self._app._open_epid_session(x, queue=queue, sessions=self._all)
        if report:
            self._app._generate_report()

    def _import_all(self):
        new = [x for x in self._all if x.status == "new"]
        if not new:
            QMessageBox.information(self, "Import", "There are no new sessions."); return
        n_ok = sum(1 for x in new if x.complete)
        if QMessageBox.question(self, "Import all new sessions",
                                f"Analyse and save {n_ok} complete session(s) of {self._machine} ({len(new) - n_ok} incomplete are left alone)?\n\n"
                                "This can take a while (a few seconds per session); you can cancel and continue later - saved sessions are not analysed again.") != QMessageBox.StandardButton.Yes:
            return
        self._app._import_all_sessions(new, self._machine)
        import wl_sources
        conn = sqlite3.connect(DB_PATH); done = wl_sources.processed_sops(conn); conn.close()
        wl_sources.classify(self._all, done)
        self._app._save_epid_mark(self._all)
        self._fill()

    def _skip(self):
        xs = self._sessions_of(self._tree.currentItem())
        if not xs:
            return
        import wl_sources
        conn = sqlite3.connect(DB_PATH)
        for x in xs:
            wl_sources.mark(conn, x.sops.values(), self._machine, "skipped")
            x.status = "skipped"
        conn.close()
        self._app._save_epid_mark(self._all)
        self._fill()


class HistoryDialog(QDialog):
    """Review the stored results of any day, week or month, and print the report for it: a draft, or - after the physicist's review - an electronically signed report with the trend lines."""

    def __init__(self, parent):
        super().__init__(parent)
        import daily_reports
        self._app = parent
        self._dr = daily_reports
        self.setWindowTitle(f"{APP_NAME} — History and reports")
        self.resize(1100, 760)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self._machine = QComboBox(); self._machine.addItems(parent._trend_machines_with_data() or list(parent._machines))
        self._kind = QComboBox(); self._kind.addItem("Day", "day"); self._kind.addItem("Week (Mon-Sun)", "week"); self._kind.addItem("Month", "month")
        self._date = QDateEdit(); self._date.setCalendarPopup(True); self._date.setDisplayFormat("yyyy-MM-dd")
        self._prev = QPushButton("◀"); self._next = QPushButton("▶"); self._latest = QPushButton("Latest")
        for wdg in (QLabel("Machine:"), self._machine, QLabel("Report:"), self._kind, QLabel("Date in the period:"), self._date, self._prev, self._next, self._latest):
            top.addWidget(wdg)
        top.addStretch()
        lay.addLayout(top)
        self._title = QLabel(""); self._title.setStyleSheet("font-size: 15px; font-weight: bold;")
        lay.addWidget(self._title)
        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(["Date / time", "Max displacement", "WL", "MLC size (Δ ref)", "Jaw size (Δ ref)", "Picket fence max / RMS", "Worst leaf", "Machine log"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectRows); self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self._table, 2)
        self._detail = QTextEdit(); self._detail.setReadOnly(True); self._detail.setMaximumHeight(170)
        lay.addWidget(self._detail)
        lay.addWidget(QLabel("Signed reports for this machine (double-click to print again):"))
        self._signed = QTableWidget(0, 6)
        self._signed.setHorizontalHeaderLabels(["No.", "Report", "Physicist", "Signed", "Sessions", "Data unchanged since signing"])
        self._signed.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._signed.setSelectionBehavior(QTableWidget.SelectRows); self._signed.setEditTriggers(QTableWidget.NoEditTriggers)
        self._signed.setMaximumHeight(150)
        lay.addWidget(self._signed)
        row = QHBoxLayout()
        b_open = QPushButton("Open selected session in the application"); b_open.clicked.connect(self._open_in_app)
        b_open.setToolTip("Re-analyse the selected session's images and show them in the main window (images, per-angle results, field sizes, picket fence), as when it was first imported. "
                          "The images are found in the machine's EPID folder; if they are not there you are asked for the folder.")
        row0 = QHBoxLayout(); row0.addWidget(b_open); row0.addStretch(); lay.addLayout(row0)
        b_draft = QPushButton("Draft report (PDF)…"); b_sign = QPushButton("Review and sign report (PDF)…"); b_close = QPushButton("Close")
        b_draft.setToolTip("An unsigned report for the period shown, saved as a PDF and opened for printing.")
        b_sign.setToolTip("The physicist confirms having reviewed the period; the PDF carries the signature, the date and time and a digest of the data, and the sign-off is recorded.")
        b_draft.clicked.connect(lambda: self._make(sign=False)); b_sign.clicked.connect(lambda: self._make(sign=True)); b_close.clicked.connect(self.reject)
        row.addWidget(b_draft); row.addWidget(b_sign); row.addStretch(); row.addWidget(b_close)
        lay.addLayout(row)
        self._machine.currentTextChanged.connect(self._go_latest)
        self._kind.currentIndexChanged.connect(self._refresh); self._date.dateChanged.connect(self._refresh)
        self._prev.clicked.connect(lambda: self._shift(-1)); self._next.clicked.connect(lambda: self._shift(1)); self._latest.clicked.connect(self._go_latest)
        self._table.itemSelectionChanged.connect(self._show_detail)
        self._table.cellDoubleClicked.connect(lambda *_: self._open_in_app())
        self._signed.cellDoubleClicked.connect(self._reprint)
        self._recs = []
        self._go_latest()

    # -- helpers
    def _limits(self):
        return {"walk": TOLERANCE_MM, "mlc": (FIELD_SIZE_WARN_MM, FIELD_SIZE_FAIL_MM), "jaw": (FIELD_SIZE_JAW_WARN_MM, FIELD_SIZE_JAW_FAIL_MM), "pf": (PF_WARN_MM, PF_TOLERANCE_MM)}

    def _period(self):
        d = self._date.date().toPython()
        return self._dr.period(self._kind.currentData(), d)

    def _go_latest(self, *_):
        days = self._dr.days_with_data(DB_PATH, self._machine.currentText())
        self._date.blockSignals(True)
        d = days[-1] if days else datetime.date.today()
        self._date.setDate(QDate(d.year, d.month, d.day))
        self._date.blockSignals(False)
        self._refresh()

    def _shift(self, n):
        a, b = self._period()
        d = (b + datetime.timedelta(days=1)) if n > 0 else (a - datetime.timedelta(days=1))
        if self._kind.currentData() == "day":
            days = self._dr.days_with_data(DB_PATH, self._machine.currentText())
            cand = [x for x in days if (x > a if n > 0 else x < a)]
            if cand:
                d = min(cand) if n > 0 else max(cand)
        self._date.setDate(QDate(d.year, d.month, d.day))

    def _refresh(self, *_):
        machine = self._machine.currentText(); kind = self._kind.currentData(); a, b = self._period()
        self._title.setText(f"{machine} — {self._dr.label(kind, a, b)}")
        self._recs = self._dr.records_between(DB_PATH, machine, a, b)
        refs = self._app._get_field_refs(machine)
        t = self._table; t.setRowCount(len(self._recs))
        lim = self._limits()
        for i, r in enumerate(self._recs):
            w_, pf = r.get("walk_circle_r"), r.get("pf")
            dm = None if r.get("avg_mlc_mm") is None else r["avg_mlc_mm"] - refs["mlc"]
            dj = None if r.get("avg_jaw_mm") is None else r["avg_jaw_mm"] - refs["jaw"]
            cells = [f"{r['when']:%Y-%m-%d %H:%M}", "—" if w_ is None else f"{w_:.3f} mm", "—" if w_ is None else "PASS" if w_ <= TOLERANCE_MM else "FAIL",
                     "—" if r.get("avg_mlc_mm") is None else f"{r['avg_mlc_mm']:.2f} ({dm:+.2f})", "—" if r.get("avg_jaw_mm") is None else f"{r['avg_jaw_mm']:.2f} ({dj:+.2f})",
                     "—" if not pf else f"{pf['max_dev_mm']:.2f} / {pf['rms_dev_mm']:.2f} mm", "—" if not pf else f"{pf['worst_leaf']} ({pf['worst_dev_mm']:+.2f})",
                     "—" if not r.get("logs_used") else f"{r['logs_used']} of 4"]
            cols = {2: None if w_ is None else ("#66bb6a" if w_ <= TOLERANCE_MM else "#ef5350"),
                    3: None if dm is None else ("#66bb6a" if abs(dm) <= FIELD_SIZE_WARN_MM else "#ffa726" if abs(dm) <= FIELD_SIZE_FAIL_MM else "#ef5350"),
                    4: None if dj is None else ("#66bb6a" if abs(dj) <= FIELD_SIZE_JAW_WARN_MM else "#ffa726" if abs(dj) <= FIELD_SIZE_JAW_FAIL_MM else "#ef5350"),
                    5: None if not pf else ("#66bb6a" if pf["max_dev_mm"] < PF_WARN_MM else "#ffa726" if pf["max_dev_mm"] < PF_TOLERANCE_MM else "#ef5350")}
            for c, txt in enumerate(cells):
                it = QTableWidgetItem(txt); it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if cols.get(c):
                    it.setForeground(QBrush(QColor(cols[c])))
                t.setItem(i, c, it)
        if self._recs:
            t.selectRow(len(self._recs) - 1)
        else:
            self._detail.setPlainText("No results were recorded in this period.")
        self._fill_signed()

    def _show_detail(self):
        r = self._table.currentRow()
        if not (0 <= r < len(self._recs)):
            return
        rec = self._recs[r]
        ir = {a: {"dx_mm": rec.get(f"raw_dx_G{a}"), "dy_mm": rec.get(f"raw_dy_G{a}")} for a in GANTRY_ANGLES}
        lines = [f"Session {rec['when']:%Y-%m-%d %H:%M}   physicist: {rec.get('physicist_name') or '—'}"]
        if all(v["dx_mm"] is not None and v["dy_mm"] is not None for v in ir.values()):
            res = compute_wl_results(ir, use_logs=False)
            lines.append(f"Static phantom offset removed: lateral {res['setup_x']:+.3f}, SI {res['setup_y']:+.3f}, AP {res['setup_z']:+.3f} mm.   "
                         f"Maximum displacement from the isocenter {res['max_2d_walk_mm']:.3f} mm ({'PASS' if res['pass_fail'] else 'FAIL'}, limit {TOLERANCE_MM:.1f} mm)")
            lines.append("Gantry    raw lateral   raw SI    after offset: lateral   SI    displacement")
            for a in GANTRY_ANGLES:
                p = res["per_angle"][a]
                lines.append(f"G{a:<3}     {p['raw_dx']:+8.3f}  {p['raw_dy']:+8.3f}      {p['rel_dx']:+8.3f}  {p['rel_dy']:+8.3f}    {np.hypot(p['rel_dx'], p['rel_dy']):.3f} mm")
        pf = rec.get("pf")
        if pf:
            lines.append(f"Picket fence: max {pf['max_dev_mm']:.3f} mm, RMS {pf['rms_dev_mm']:.3f} mm, worst leaf {pf['worst_leaf']} ({pf['worst_dev_mm']:+.3f} mm), tilt along the picket "
                         + ("—" if pf.get("tilt") is None else f"{pf['tilt']:+.2f} mm"))
        self._detail.setPlainText("\n".join(lines))

    def _open_in_app(self):
        """Load the selected stored session into the main window by re-analysing its images."""
        r = self._table.currentRow()
        if not (0 <= r < len(self._recs)):
            QMessageBox.information(self, "Open session", "Select a session in the list first."); return
        rec, machine = self._recs[r], self._machine.currentText()
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            sess = self._app._find_session_images(machine, rec["when"])
        finally:
            self.unsetCursor()
        if sess is None:
            QMessageBox.information(self, "Images not found",
                                    f"The images of the {rec['when']:%Y-%m-%d %H:%M} session were not found in the EPID folder of {machine}.\n\n"
                                    "Choose the folder that holds them (the session folder, or any folder above it).")
            folder = QFileDialog.getExistingDirectory(self, "Folder with the session's images", self._app._config.get("last_dicom_dir", str(Path.home())))
            if not folder:
                return
            self.accept()
            self._app._machine_combo.setCurrentText(machine)
            self._app._process_directory(folder, f"{machine}: {rec['when']:%Y-%m-%d %H:%M}")
            return
        self.accept()
        self._app._open_epid_session(sess)

    def _fill_signed(self):
        rows = self._dr.signoffs(DB_PATH, self._machine.currentText())
        self._signed_rows = rows
        self._signed.setRowCount(len(rows))
        for i, so in enumerate(rows):
            a, b = datetime.date.fromisoformat(so["start_date"]), datetime.date.fromisoformat(so["end_date"])
            ok = self._dr.still_matches(DB_PATH, so)
            for c, txt in enumerate((str(so["id"]), f"{so['kind']}: {self._dr.label(so['kind'], a, b)}", so["physicist"], so["signed_at"].replace("T", " "), str(so["n_sessions"]),
                                     "yes" if ok else "NO - records changed")):
                it = QTableWidgetItem(txt)
                if c == 5:
                    it.setForeground(QBrush(QColor("#66bb6a" if ok else "#ef5350")))
                self._signed.setItem(i, c, it)

    # -- reports
    def _write(self, path, kind, a, b, machine, signoff=None):
        self._dr.build_report(DB_PATH, machine, kind, a, b, t=sys.modules[__name__], refs=self._app._get_field_refs(machine), limits=self._limits(), version=APP_VERSION, signoff=signoff, out_path=path)

    def _ask_path(self, name):
        last = self._app._config.get("last_report_dir", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(self, "Save report", str(Path(last) / name), "PDF files (*.pdf)")
        if path:
            self._app._config["last_report_dir"] = str(Path(path).parent); _save_config(self._app._config)
        return path

    def _make(self, sign: bool):
        machine = self._machine.currentText(); kind = self._kind.currentData(); a, b = self._period()
        if not self._recs:
            QMessageBox.information(self, "Report", "There are no results in this period."); return
        stem = f"DailyQA_{machine.replace(' ', '_')}_{kind}_{a:%Y%m%d}" + ("" if kind == "day" else f"-{b:%Y%m%d}")
        physicist = self._app._physicist_combo.currentText()
        comment = ""
        if sign:
            if QMessageBox.question(self, "Sign the report",
                                    f"{physicist}\n\nconfirms having reviewed the {len(self._recs)} session(s) of {self._dr.label(kind, a, b)} on {machine}, and signs this report electronically?\n\n"
                                    "(The physicist is the one chosen in the main window.)") != QMessageBox.StandardButton.Yes:
                return
            comment, ok = QInputDialog.getText(self, "Comment", "Comment for the report (optional):")
            if not ok:
                return
        path = self._ask_path(stem + ("_signed.pdf" if sign else "_draft.pdf"))
        if not path:
            return
        try:
            so = self._dr.add_signoff(DB_PATH, machine, kind, a, b, physicist, comment, self._recs, Path(path).name) if sign else None
            self._write(path, kind, a, b, machine, so)
        except Exception as exc:
            QMessageBox.critical(self, "Report error", f"{type(exc).__name__}: {exc}"); return
        self._fill_signed()
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))              # opens in the PDF viewer, from where it is printed

    def _reprint(self, row, _col):
        if not (0 <= row < len(self._signed_rows)):
            return
        so = self._signed_rows[row]
        a, b = datetime.date.fromisoformat(so["start_date"]), datetime.date.fromisoformat(so["end_date"])
        if not self._dr.still_matches(DB_PATH, so):
            if QMessageBox.question(self, "Data changed", "The records of this period have changed since the report was signed, so the report printed now would not match what was signed.\n\n"
                                    "Print it anyway (it will show the signed digest, which will no longer match)?") != QMessageBox.StandardButton.Yes:
                return
        path = self._ask_path(f"DailyQA_{so['machine_name'].replace(' ', '_')}_{so['kind']}_{a:%Y%m%d}_signed_{so['id']}.pdf")
        if not path:
            return
        meta = {"id": so["id"], "physicist": so["physicist"], "signed_at": so["signed_at"], "comment": so["comment"], "n": so["n_sessions"], "digest": so["digest"]}
        self._write(path, so["kind"], a, b, so["machine_name"], meta)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


class WLApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}  —  Elekta Versa HD / MIMI Phantom")
        self.resize(1160, 860)
        self.setMinimumSize(960, 720)

        for icon_name in ("icon.ico", "icon.png"):
            _icon_path = Path(__file__).parent / icon_name
            if _icon_path.exists():
                self.setWindowIcon(QIcon(str(_icon_path)))
                break

        self._wl_results    = None
        self._image_results = None
        self._loaded_dir    = None
        self._diag_fig_path = None
        self._walk_fig_path = None
        self._dicom_date    = None
        self._dicom_time    = None
        self._pf_results    = None   # picket fence results (None if not performed today)
        self._pf_fig_path   = None
        self._table_labels: dict  = {}
        self._res_ref_labels: dict = {}

        self._config = _load_config()
        self._machines   = list(self._config.get("machines", MACHINES))
        self._physicists = list(self._config.get("physicists", PHYSICISTS))
        _init_db()
        self._build_ui()
        # Show saved calibration values (or first-use notice) immediately on startup
        self._update_field_ref_labels()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 18, 20, 16)
        main_layout.setSpacing(6)

        # ── Top bar ──────────────────────────────────────────────────────────
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)

        title_lbl = QLabel(APP_NAME)
        title_lbl.setStyleSheet("font-size: 24px; font-weight: bold;")
        top_layout.addWidget(title_lbl)
        top_layout.addStretch()

        self._dir_label = QLabel("No directory loaded")
        self._dir_label.setStyleSheet("font-size: 13px; color: gray;")
        top_layout.addWidget(self._dir_label)

        self._load_btn = QPushButton("Load DICOM Directory")
        self._load_btn.setMinimumWidth(210)
        self._load_btn.clicked.connect(self._load_directory)
        top_layout.addWidget(self._load_btn)

        self._epid_btn = QPushButton("Check EPID Folder")
        self._epid_btn.setMinimumWidth(180)
        self._epid_btn.setToolTip("Search this machine's EPID folder (Settings > Data Folders) for Daily WL sessions not processed yet")
        self._epid_btn.clicked.connect(lambda: self._check_epid_folder())
        top_layout.addWidget(self._epid_btn)
        self._epid_full_btn = QPushButton("Full Rescan")
        self._epid_full_btn.setToolTip("Read the whole EPID folder again, not only what is newer than the last check (slow on a network drive; only needed if images were added out of order).")
        self._epid_full_btn.clicked.connect(lambda: self._check_epid_folder(full=True))
        top_layout.addWidget(self._epid_full_btn)

        main_layout.addWidget(top_bar)

        self._queue, self._queue_cur, self._queue_all = [], None, None
        self._queue_bar = QWidget()
        qb = QHBoxLayout(self._queue_bar); qb.setContentsMargins(0, 0, 0, 0)
        self._queue_label = QLabel(""); self._queue_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4fc3f7;")
        self._queue_prev = QPushButton("◀ Previous session"); self._queue_next = QPushButton("Review next session ▶"); b_list = QPushButton("Back to the list")
        self._queue_prev.clicked.connect(lambda: self._queue_step(-1)); self._queue_next.clicked.connect(lambda: self._queue_step(1)); b_list.clicked.connect(self._queue_list)
        for wdg in (self._queue_label,):
            qb.addWidget(wdg)
        qb.addStretch()
        for wdg in (self._queue_prev, self._queue_next, b_list):
            qb.addWidget(wdg)
        self._queue_bar.setVisible(False)
        main_layout.addWidget(self._queue_bar)

        # ── Selector bar ──────────────────────────────────────────────────────
        sel_bar = QWidget()
        sel_layout = QHBoxLayout(sel_bar)
        sel_layout.setContentsMargins(0, 0, 0, 0)

        machine_lbl = QLabel("Machine:")
        machine_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        sel_layout.addWidget(machine_lbl)

        self._machine_combo = QComboBox()
        self._machine_combo.addItems(self._machines)
        self._machine_combo.setMinimumWidth(240)
        self._machine_combo.currentIndexChanged.connect(self._update_field_ref_labels)
        sel_layout.addWidget(self._machine_combo)
        sel_layout.addSpacing(24)

        physicist_lbl = QLabel("Physicist:")
        physicist_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        sel_layout.addWidget(physicist_lbl)

        self._physicist_combo = QComboBox()
        self._physicist_combo.addItems(self._physicists)
        self._physicist_combo.setMinimumWidth(300)
        sel_layout.addWidget(self._physicist_combo)
        sel_layout.addSpacing(24)

        self._use_logs_cb = QCheckBox("Use machine log (when present)")
        self._use_logs_cb.setToolTip("Correct each field centre for the aperture (MLC / jaw position) the machine log reports for that image. "
                                     "Works without logs: images with no matching log are analysed as measured.")
        self._use_logs_cb.setChecked(bool(self._config.get("use_machine_logs", True)))
        self._use_logs_cb.toggled.connect(self._logs_toggled)
        sel_layout.addWidget(self._use_logs_cb)
        sel_layout.addStretch()

        self._settings_btn = QPushButton("Settings")
        self._settings_btn.setMinimumWidth(110)
        self._settings_btn.clicked.connect(self._show_settings)
        sel_layout.addWidget(self._settings_btn)

        main_layout.addWidget(sel_bar)

        # ── PASS / FAIL banner  (WL left | Field right) ──────────────────────
        pf_container = QWidget()
        pf_container.setFixedHeight(72)
        pf_hbox = QHBoxLayout(pf_container)
        pf_hbox.setContentsMargins(0, 0, 0, 0)
        pf_hbox.setSpacing(8)

        self._pf_frame = QFrame()
        self._pf_frame.setStyleSheet(
            "QFrame { background-color: #2b2b2b; border-radius: 10px; }"
        )
        pf_inner_wl = QHBoxLayout(self._pf_frame)
        pf_inner_wl.setContentsMargins(0, 0, 0, 0)
        self._pf_label = QLabel("—  WL AWAITING DATA  —")
        self._pf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pf_label.setStyleSheet(
            "font-size: 25px; font-weight: bold; color: gray; background: transparent;"
        )
        pf_inner_wl.addWidget(self._pf_label)
        pf_hbox.addWidget(self._pf_frame, stretch=1)

        self._pf_frame_fs = QFrame()
        self._pf_frame_fs.setStyleSheet(
            "QFrame { background-color: #2b2b2b; border-radius: 10px; }"
        )
        pf_inner_fs = QHBoxLayout(self._pf_frame_fs)
        pf_inner_fs.setContentsMargins(0, 0, 0, 0)
        self._pf_label_fs = QLabel("—  FIELD AWAITING DATA  —")
        self._pf_label_fs.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pf_label_fs.setStyleSheet(
            "font-size: 25px; font-weight: bold; color: gray; background: transparent;"
        )
        pf_inner_fs.addWidget(self._pf_label_fs)
        pf_hbox.addWidget(self._pf_frame_fs, stretch=1)

        self._pf_frame_picket = QFrame()
        self._pf_frame_picket.setStyleSheet(
            "QFrame { background-color: #2b2b2b; border-radius: 10px; }"
        )
        pf_inner_picket = QHBoxLayout(self._pf_frame_picket)
        pf_inner_picket.setContentsMargins(0, 0, 0, 0)
        self._pf_label_picket = QLabel("—  PF AWAITING DATA  —")
        self._pf_label_picket.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pf_label_picket.setStyleSheet(
            "font-size: 25px; font-weight: bold; color: gray; background: transparent;"
        )
        pf_inner_picket.addWidget(self._pf_label_picket)
        pf_hbox.addWidget(self._pf_frame_picket, stretch=1)
        self._pf_frame_picket.setVisible(False)   # shown only when PF data loaded

        main_layout.addWidget(pf_container)

        # ── Tab view ──────────────────────────────────────────────────────────
        self._tabview = QTabWidget()
        self._tabview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(self._tabview, stretch=1)

        # Results tab ─────────────────────────────────────────────────────────
        tab_res = QWidget()
        res_layout = QHBoxLayout(tab_res)
        res_layout.setContentsMargins(6, 6, 6, 6)
        res_layout.setSpacing(10)
        self._tabview.addTab(tab_res, "Results")

        # ── Left panel: WL tables stacked vertically ──────────────────────────
        wl_panel = QWidget()
        wl_panel.setStyleSheet("background: transparent;")
        wl_vbox = QVBoxLayout(wl_panel)
        wl_vbox.setContentsMargins(0, 0, 0, 0)
        wl_vbox.setSpacing(8)

        corr_card = QFrame()
        corr_card.setStyleSheet("QFrame { background-color: #363636; border-radius: 10px; }")
        corr_layout = QVBoxLayout(corr_card)
        corr_layout.setContentsMargins(12, 10, 12, 8)
        corr_title = QLabel("Corrected Displacements  (Isocenter Walk)")
        corr_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        corr_title.setStyleSheet("font-size: 14px; font-weight: bold; background: transparent;")
        corr_layout.addWidget(corr_title)
        self._corr_sub = QLabel("Isocenter walk  (3D setup error removed)")
        self._corr_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._corr_sub.setStyleSheet("font-size: 11px; color: gray; background: transparent;")
        corr_layout.addWidget(self._corr_sub)
        corr_tbl_widget = QWidget()
        corr_tbl_widget.setStyleSheet("background: transparent;")
        self._build_result_table(corr_tbl_widget, prefix="corr")
        corr_layout.addWidget(corr_tbl_widget, stretch=1)
        wl_vbox.addWidget(corr_card, stretch=1)

        # Walk circle displacement map
        walk_fig_frame = QFrame()
        walk_fig_frame.setStyleSheet(
            "QFrame { background-color: #1a1a1a; border-radius: 10px; }"
        )
        walk_fig_inner = QVBoxLayout(walk_fig_frame)
        walk_fig_inner.setContentsMargins(4, 4, 4, 4)
        self._walk_fig_label = QLabel()
        self._walk_fig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._walk_fig_label.setStyleSheet("background: transparent;")
        walk_fig_inner.addWidget(self._walk_fig_label)
        wl_vbox.addWidget(walk_fig_frame, stretch=2)

        res_layout.addWidget(wl_panel, stretch=4)

        # ── Right panel: Field size data ──────────────────────────────────────
        fs_panel = QFrame()
        fs_panel.setStyleSheet("QFrame { background-color: #363636; border-radius: 10px; }")
        fs_vbox = QVBoxLayout(fs_panel)
        fs_vbox.setContentsMargins(12, 10, 12, 10)
        fs_vbox.setSpacing(5)

        fs_ref_row = QWidget()
        fs_ref_row.setStyleSheet("background: transparent;")
        fs_ref_hbox = QHBoxLayout(fs_ref_row)
        fs_ref_hbox.setContentsMargins(0, 0, 0, 0)
        fs_ref_hbox.setSpacing(6)
        ref_hdr = QLabel("Baseline Reference:")
        ref_hdr.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #aaaaaa; background: transparent;"
        )
        fs_ref_hbox.addWidget(ref_hdr)
        for _rkey, _rdisp in [("mlc", "  MLC (Leaves):"), ("jaw", "  Jaw (SI):")]:
            nlbl = QLabel(_rdisp)
            nlbl.setStyleSheet("font-size: 12px; color: #777777; background: transparent;")
            fs_ref_hbox.addWidget(nlbl)
            vlbl = QLabel("—")
            vlbl.setStyleSheet(
                "font-size: 13px; font-family: monospace; font-weight: bold; "
                "color: #90caf9; background: transparent;"
            )
            fs_ref_hbox.addWidget(vlbl)
            self._res_ref_labels[_rkey] = vlbl
        fs_ref_hbox.addStretch()
        fs_vbox.addWidget(fs_ref_row)

        ang_hdr_lbl2 = QLabel(
            f"Field Size per Gantry Angle   —   "
            f"MLC: Warn >{FIELD_SIZE_WARN_MM:.1f} mm  Fail >{FIELD_SIZE_FAIL_MM:.1f} mm   —   "
            f"Jaw: Warn >{FIELD_SIZE_JAW_WARN_MM:.1f} mm  Fail >{FIELD_SIZE_JAW_FAIL_MM:.1f} mm"
        )
        ang_hdr_lbl2.setStyleSheet("font-size: 11px; color: #666666;")
        fs_vbox.addWidget(ang_hdr_lbl2)

        res_ang_hdrs = ["Angle", "MLC (mm)", "Δ MLC (mm)", "Jaw (mm)", "Δ Jaw (mm)"]
        self._res_fs_angle_table = QTableWidget(5, len(res_ang_hdrs))
        self._res_fs_angle_table.setHorizontalHeaderLabels(res_ang_hdrs)
        self._res_fs_angle_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._res_fs_angle_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._res_fs_angle_table.verticalHeader().setVisible(False)
        self._res_fs_angle_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._res_fs_angle_table.setMaximumHeight(185)
        for _ri, _albl in enumerate([f"G{a:03d}°" for a in GANTRY_ANGLES] + ["Mean"]):
            _it = QTableWidgetItem(_albl)
            _it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            _it.setForeground(QBrush(QColor("#90caf9")))
            self._res_fs_angle_table.setItem(_ri, 0, _it)
        fs_vbox.addWidget(self._res_fs_angle_table)

        leaf_hdr_lbl2 = QLabel(
            f"Individual MLC Leaf Spans   —   "
            f"{MLC_LEAVES_PER_BANK} leaves × {MLC_LEAF_WIDTH_MM:.0f} mm each (SI)   —   G000°"
        )
        leaf_hdr_lbl2.setStyleSheet("font-size: 11px; color: #666666;")
        fs_vbox.addWidget(leaf_hdr_lbl2)

        res_leaf_hdrs = ["Leaf #", "SI Centre (mm)", "Total (mm)", "Δ Span (mm)"]
        self._res_fs_leaf_table = QTableWidget(MLC_LEAVES_PER_BANK, len(res_leaf_hdrs))
        self._res_fs_leaf_table.setHorizontalHeaderLabels(res_leaf_hdrs)
        self._res_fs_leaf_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._res_fs_leaf_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._res_fs_leaf_table.verticalHeader().setVisible(False)
        self._res_fs_leaf_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        for _ri in range(MLC_LEAVES_PER_BANK):
            _si = (MLC_LEAVES_PER_BANK / 2 - _ri - 0.5) * MLC_LEAF_WIDTH_MM
            for _ci, (_txt, _col) in enumerate([
                (str(_ri + 1),     "#90caf9"),
                (f"{_si:+.1f}",   "#aaaaaa"),
            ]):
                _it = QTableWidgetItem(_txt)
                _it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                _it.setForeground(QBrush(QColor(_col)))
                self._res_fs_leaf_table.setItem(_ri, _ci, _it)
        fs_vbox.addWidget(self._res_fs_leaf_table, stretch=1)

        res_layout.addWidget(fs_panel, stretch=5)

        # Field Size QA tab ───────────────────────────────────────────────────
        self._build_field_size_tab()
        self._build_pf_tab()

        # Portal Images tab ───────────────────────────────────────────────────
        tab_img = QWidget()
        img_layout = QVBoxLayout(tab_img)
        img_layout.setContentsMargins(6, 6, 6, 6)
        self._tabview.addTab(tab_img, "Portal Images")

        self._diag_img_label = QLabel(
            "Load a DICOM directory to view portal images"
        )
        self._diag_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._diag_img_label.setStyleSheet("font-size: 14px; color: #888888;")

        img_scroll = QScrollArea()
        img_scroll.setWidgetResizable(True)
        img_scroll.setWidget(self._diag_img_label)
        img_layout.addWidget(img_scroll)

        # ── Bottom bar ────────────────────────────────────────────────────────
        bottom_bar = QWidget()
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0, 4, 0, 0)

        # Stack WL walk label + DLG status label on the left
        left_status = QWidget()
        left_status.setStyleSheet("background: transparent;")
        left_vbox = QVBoxLayout(left_status)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(2)

        self._walk_label = QLabel(
            f"Max displacement: —   (Tolerance ≤ {TOLERANCE_MM:.1f} mm)"
        )
        self._walk_label.setStyleSheet("font-size: 15px;")
        left_vbox.addWidget(self._walk_label)

        self._dlg_status_label = QLabel("DLG Field Size:  —")
        self._dlg_status_label.setStyleSheet("font-size: 13px; color: #888888;")
        left_vbox.addWidget(self._dlg_status_label)

        bottom_layout.addWidget(left_status)
        bottom_layout.addStretch()

        self._trends_btn = QPushButton("View Trends")
        self._trends_btn.setMinimumWidth(150)
        self._trends_btn.clicked.connect(self._show_trends)
        bottom_layout.addWidget(self._trends_btn)

        self._history_btn = QPushButton("History / Reports")
        self._history_btn.setMinimumWidth(150)
        self._history_btn.setToolTip("Review the results of any day, week or month and print the report (draft, or signed by the physicist).")
        self._history_btn.clicked.connect(lambda: HistoryDialog(self).exec())
        bottom_layout.addWidget(self._history_btn)

        self._report_btn = QPushButton("Generate Daily Report (PDF)")
        self._report_btn.setMinimumWidth(230)
        self._report_btn.setEnabled(False)
        self._report_btn.clicked.connect(self._generate_report)
        bottom_layout.addWidget(self._report_btn)

        main_layout.addWidget(bottom_bar)

    def _build_result_table(self, parent: QWidget, prefix: str):
        grid = QGridLayout(parent)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(4)

        headers = ["Angle", "ΔLat / ΔAP (mm)", "ΔSI (mm)", "|ΔR| (mm)"]
        for c, h in enumerate(headers):
            grid.setColumnStretch(c, 1)
            lbl = QLabel(h)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                "color: #aaaaaa; font-size: 15px; font-weight: bold;"
            )
            grid.addWidget(lbl, 0, c)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("QFrame { background-color: #3a3a3a; }")
        grid.addWidget(sep, 1, 0, 1, 4)

        for i, angle in enumerate(GANTRY_ANGLES):
            row = i + 2
            x_dir = "AP" if angle in (90, 270) else "Lat"
            angle_lbl = QLabel(f"G{angle:03d}°  {x_dir}")
            angle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            angle_lbl.setStyleSheet("font-size: 17px; font-weight: bold;")
            grid.addWidget(angle_lbl, row, 0)

            for c, suffix in enumerate(["dx", "dy", "dr"], start=1):
                key = f"{prefix}_{angle}_{suffix}"
                lbl = QLabel("—")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet(
                    "font-size: 17px; font-family: monospace; color: #dcdcdc;"
                )
                grid.addWidget(lbl, row, c)
                self._table_labels[key] = lbl

        grid.setRowStretch(len(GANTRY_ANGLES) + 2, 1)

    def _build_field_size_tab(self):
        """Build the Field Size QA tab — total MLC and Jaw field sizes only."""
        tab_fs = QWidget()
        fs_layout = QVBoxLayout(tab_fs)
        fs_layout.setContentsMargins(10, 10, 10, 10)
        fs_layout.setSpacing(10)
        self._tabview.addTab(tab_fs, "Field Size QA")

        # ── Reference row ─────────────────────────────────────────────────────
        ref_frame = QFrame()
        ref_frame.setStyleSheet(
            "QFrame { background-color: #363636; border-radius: 8px; }"
        )
        ref_inner = QHBoxLayout(ref_frame)
        ref_inner.setContentsMargins(12, 8, 12, 8)

        ref_title = QLabel("Reference:")
        ref_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; background: transparent;"
        )
        ref_inner.addWidget(ref_title)
        ref_inner.addSpacing(16)

        self._fs_ref_labels: dict = {}
        for key, display in [("mlc", "MLC (Leaves)"), ("jaw", "Jaw (SI)")]:
            lbl_name = QLabel(f"{display}:")
            lbl_name.setStyleSheet(
                "font-size: 13px; color: #aaaaaa; background: transparent;"
            )
            ref_inner.addWidget(lbl_name)
            lbl_val = QLabel(f"{FIELD_SIZE_REF_MM:.3f} mm")
            lbl_val.setStyleSheet(
                "font-size: 14px; font-family: monospace; font-weight: bold; "
                "color: #90caf9; background: transparent; margin-right: 20px;"
            )
            ref_inner.addWidget(lbl_val)
            self._fs_ref_labels[key] = lbl_val

        ref_inner.addStretch()
        set_ref_btn = QPushButton("Set Current as Reference")
        set_ref_btn.setMinimumWidth(230)
        set_ref_btn.clicked.connect(self._set_field_size_reference)
        ref_inner.addWidget(set_ref_btn)
        fs_layout.addWidget(ref_frame)

        # ── First-use notice (hidden once a reference is saved) ───────────────
        self._fs_notice_label = QLabel(
            "⚠  First-time setup: load images, then click  'Set Current as Reference'  "
            "to record this machine's baseline field size.  "
            "PASS / FAIL comparisons are not shown until a reference is saved."
        )
        self._fs_notice_label.setStyleSheet(
            "font-size: 12px; color: #ffa726; padding: 6px 12px; "
            "background: #3e2c00; border-radius: 6px;"
        )
        self._fs_notice_label.setWordWrap(True)
        fs_layout.addWidget(self._fs_notice_label)

        # ── Per-angle table ───────────────────────────────────────────────────
        ang_hdr_lbl = QLabel(
            "Field Size per Gantry Angle  "
            "— MLC leaves (Lat at G0°/G180°, AP at G90°/G270°)  "
            f"[Warn >{FIELD_SIZE_WARN_MM:.1f} mm, Call Service >{FIELD_SIZE_FAIL_MM:.1f} mm]  "
            "— Jaw (always SI)  "
            f"[Warn >{FIELD_SIZE_JAW_WARN_MM:.1f} mm, Call Service >{FIELD_SIZE_JAW_FAIL_MM:.1f} mm]"
        )
        ang_hdr_lbl.setStyleSheet("font-size: 12px; color: #888888;")
        fs_layout.addWidget(ang_hdr_lbl)

        # 3-column table: just deviations from reference (no raw totals cluttering display)
        angle_hdrs = ["Angle", "Δ MLC (mm)", "Δ Jaw (mm)"]
        self._fs_angle_table = QTableWidget(5, len(angle_hdrs))
        self._fs_angle_table.setHorizontalHeaderLabels(angle_hdrs)
        self._fs_angle_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._fs_angle_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._fs_angle_table.verticalHeader().setVisible(False)
        self._fs_angle_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._fs_angle_table.setMaximumHeight(210)
        angle_row_labels = [f"G{a:03d}°" for a in GANTRY_ANGLES] + ["Mean"]
        for ri, lbl in enumerate(angle_row_labels):
            item = QTableWidgetItem(lbl)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setForeground(QBrush(QColor("#90caf9")))
            self._fs_angle_table.setItem(ri, 0, item)
        fs_layout.addWidget(self._fs_angle_table)

        # ── MLC leaf table ────────────────────────────────────────────────────
        leaf_hdr_lbl = QLabel(
            f"Individual MLC Leaf Spans  "
            f"— {MLC_LEAVES_PER_BANK} leaves × {MLC_LEAF_WIDTH_MM:.0f} mm each (SI)  "
            f"— from G000° image  — deviation from leaf baseline"
        )
        leaf_hdr_lbl.setStyleSheet("font-size: 12px; color: #888888;")
        fs_layout.addWidget(leaf_hdr_lbl)

        leaf_hdrs = ["Leaf #", "SI Centre (mm)", "Δ Span (mm)"]
        self._fs_leaf_table = QTableWidget(MLC_LEAVES_PER_BANK, len(leaf_hdrs))
        self._fs_leaf_table.setHorizontalHeaderLabels(leaf_hdrs)
        self._fs_leaf_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._fs_leaf_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._fs_leaf_table.verticalHeader().setVisible(False)
        self._fs_leaf_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        for ri in range(MLC_LEAVES_PER_BANK):
            # Leaf 0 = top (most superior), so SI centre goes from +ve to -ve
            si_pos = (MLC_LEAVES_PER_BANK / 2 - ri - 0.5) * MLC_LEAF_WIDTH_MM
            for ci, (txt, col) in enumerate([
                (str(ri + 1),      "#90caf9"),
                (f"{si_pos:+.1f}", "#aaaaaa"),
            ]):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setForeground(QBrush(QColor(col)))
                self._fs_leaf_table.setItem(ri, ci, item)
        fs_layout.addWidget(self._fs_leaf_table, stretch=1)

    def _build_pf_tab(self):
        """Build the Picket Fence tab (shown always; content appears after PF load)."""
        tab_pf = QWidget()
        pf_layout = QVBoxLayout(tab_pf)
        pf_layout.setContentsMargins(6, 6, 6, 6)
        pf_layout.setSpacing(6)
        self._tabview.addTab(tab_pf, "Picket Fence")

        # Stats card
        self._pf_stats_card = QFrame()
        self._pf_stats_card.setStyleSheet(
            "QFrame { background-color: #363636; border-radius: 8px; }"
        )
        stats_layout = QHBoxLayout(self._pf_stats_card)
        stats_layout.setContentsMargins(16, 8, 16, 8)
        stats_layout.setSpacing(30)

        for key, label_text in [
            ("_pf_stat_leaves",  "Leaves"),
            ("_pf_stat_max",     "Max |Δ| (mm)"),
            ("_pf_stat_rms",     "RMS (mm)"),
            ("_pf_stat_width",   "Avg Width (mm)"),
        ]:
            col = QWidget()
            col.setStyleSheet("background: transparent;")
            col_v = QVBoxLayout(col)
            col_v.setContentsMargins(0, 0, 0, 0)
            col_v.setSpacing(2)
            name_lbl = QLabel(label_text)
            name_lbl.setStyleSheet("color:#aaaaaa; font-size:12px; background:transparent;")
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_lbl = QLabel("—")
            val_lbl.setStyleSheet("color:#dcdcdc; font-size:17px; font-weight:bold; "
                                  "font-family:monospace; background:transparent;")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col_v.addWidget(name_lbl)
            col_v.addWidget(val_lbl)
            stats_layout.addWidget(col)
            setattr(self, key, val_lbl)

        stats_layout.addStretch()
        self._pf_stats_card.setVisible(False)
        pf_layout.addWidget(self._pf_stats_card)

        # Placeholder / figure area
        self._pf_img_label = QLabel(
            "Load a Picket Fence DICOM directory or file to view results.\n"
            "PF images in the same WL directory are detected automatically."
        )
        self._pf_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pf_img_label.setStyleSheet("font-size: 14px; color: #888888;")

        pf_scroll = QScrollArea()
        pf_scroll.setWidgetResizable(True)
        pf_scroll.setWidget(self._pf_img_label)
        pf_layout.addWidget(pf_scroll, stretch=1)

    def _get_field_refs(self, machine_name: str) -> dict:
        """Return per-machine field size reference values.

        Returns is_calibrated=False when no machine-specific reference has been
        saved yet — callers should suppress PASS/WARN/FAIL in that state.
        'leaf' is stored separately from 'mlc' because individual leaf spans
        measured in 5 mm strips are systematically smaller than the full-field
        total due to narrower profile averaging.
        """
        stored = self._config.get("field_refs", {}).get(machine_name, {})
        calibrated = bool(stored)
        ref_mlc = stored.get("mlc", FIELD_SIZE_REF_MM)
        return {
            "mlc":           ref_mlc,
            "jaw":           stored.get("jaw", FIELD_SIZE_REF_MM),
            "leaf":          stored.get("leaf", ref_mlc),  # falls back to mlc if old config
            "is_calibrated": calibrated,
        }

    def _set_field_size_reference(self):
        """Save mean of current measurements as the reference for the selected machine."""
        if self._image_results is None:
            QMessageBox.warning(self, "No Data", "Load DICOM images first.")
            return
        machine = self._machine_combo.currentText()
        mlc_vals, jaw_vals, leaf_vals = [], [], []
        for ir in self._image_results.values():
            fe = ir.get("field_edges", {})
            m = fe.get("mlc_total_mm")
            j = fe.get("jaw_total_mm")
            if m is not None:
                mlc_vals.append(m)
            if j is not None:
                jaw_vals.append(j)
        # Leaf reference from G000° only (same image used in leaf table)
        for leaf in self._image_results.get(0, {}).get("mlc_leaves", []):
            t = leaf.get("total_mm")
            if t is not None:
                leaf_vals.append(t)
        new_ref = {}
        if mlc_vals:
            new_ref["mlc"]  = float(np.mean(mlc_vals))
        if jaw_vals:
            new_ref["jaw"]  = float(np.mean(jaw_vals))
        if leaf_vals:
            new_ref["leaf"] = float(np.mean(leaf_vals))
        if not new_ref:
            QMessageBox.warning(self, "No Data", "No field size measurements available.")
            return
        if "field_refs" not in self._config:
            self._config["field_refs"] = {}
        self._config["field_refs"][machine] = new_ref
        _save_config(self._config)
        self._refresh_field_size_display()
        detail = (f"MLC={new_ref.get('mlc', 0):.3f} mm,  "
                  f"Jaw={new_ref.get('jaw', 0):.3f} mm,  "
                  f"Leaf span={new_ref.get('leaf', 0):.3f} mm")
        QMessageBox.information(
            self, "Reference Saved",
            f"Field size reference for {machine} updated:\n{detail}"
        )

    def _update_field_ref_labels(self):
        """Refresh the reference labels and first-use notice from the saved config.

        Safe to call at startup before any images are loaded.
        """
        machine = self._machine_combo.currentText()
        refs = self._get_field_refs(machine)
        is_calibrated = refs["is_calibrated"]
        ref_mlc, ref_jaw = refs["mlc"], refs["jaw"]

        if is_calibrated:
            ref_style = ("font-size: 14px; font-family: monospace; font-weight: bold; "
                         "color: #90caf9; background: transparent; margin-right: 20px;")
            self._fs_ref_labels["mlc"].setText(f"{ref_mlc:.3f} mm")
            self._fs_ref_labels["jaw"].setText(f"{ref_jaw:.3f} mm")
        else:
            ref_style = ("font-size: 14px; font-family: monospace; font-weight: bold; "
                         "color: #ffa726; background: transparent; margin-right: 20px;")
            self._fs_ref_labels["mlc"].setText(f"{ref_mlc:.3f} mm  (nominal — not set)")
            self._fs_ref_labels["jaw"].setText(f"{ref_jaw:.3f} mm  (nominal — not set)")
        for lbl in self._fs_ref_labels.values():
            lbl.setStyleSheet(ref_style)
        self._fs_notice_label.setVisible(not is_calibrated)

        # Mirror into Results tab reference labels
        res_style = (
            "font-size: 13px; font-family: monospace; font-weight: bold; "
            "color: #90caf9; background: transparent;"
        ) if is_calibrated else (
            "font-size: 13px; font-family: monospace; font-weight: bold; "
            "color: #ffa726; background: transparent;"
        )
        for key, val_text in [
            ("mlc", f"{ref_mlc:.3f} mm" if is_calibrated else f"{ref_mlc:.3f} mm (nominal)"),
            ("jaw", f"{ref_jaw:.3f} mm" if is_calibrated else f"{ref_jaw:.3f} mm (nominal)"),
        ]:
            if key in self._res_ref_labels:
                self._res_ref_labels[key].setText(val_text)
                self._res_ref_labels[key].setStyleSheet(res_style)

    def _refresh_field_size_display(self):
        """Populate the Field Size QA tab from the latest image results."""
        machine = self._machine_combo.currentText()
        refs = self._get_field_refs(machine)
        ref_mlc       = refs["mlc"]
        ref_jaw       = refs["jaw"]
        ref_leaf      = refs["leaf"]
        is_calibrated = refs["is_calibrated"]

        # Always update reference labels / notice banner (safe before images are loaded)
        self._update_field_ref_labels()

        if self._image_results is None:
            return

        from PySide6.QtGui import QFont as _QFont

        def _fs_level(dev: float, kind: str = "mlc") -> int:
            """0=pass, 1=warn, 2=fail. kind='jaw' uses the wider physical-jaw tolerance."""
            a = abs(dev)
            warn, fail = (FIELD_SIZE_JAW_WARN_MM, FIELD_SIZE_JAW_FAIL_MM) if kind == "jaw" \
                else (FIELD_SIZE_WARN_MM, FIELD_SIZE_FAIL_MM)
            if a <= warn:  return 0
            if a <= fail:  return 1
            return 2

        _FS_COLORS = ["#66bb6a", "#ffd600", "#ef5350"]   # pass / warn(yellow) / fail(red)

        def _cell(text: str, color: str = "#dcdcdc", bold: bool = False) -> QTableWidgetItem:
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setForeground(QBrush(QColor(color)))
            if bold:
                f = _QFont(); f.setBold(True)
                item.setFont(f)
            return item

        def _dfmt(v, ref):  return f"{v - ref:+.3f}" if v is not None else "—"

        def _vcell_dev(v, ref, kind: str = "mlc"):
            """Δ from reference cell — grey dash when reference not yet calibrated."""
            if not is_calibrated:
                return _cell("—", "#555555")
            dev   = (v - ref) if v is not None else None
            lvl   = _fs_level(dev, kind) if dev is not None else -1
            color = _FS_COLORS[lvl] if lvl >= 0 else "#888888"
            bold  = (lvl == 2)
            return _cell(_dfmt(v, ref), color, bold)

        mlc_acc, jaw_acc = [], []
        mlc_devs, jaw_devs = [], []
        worst_lvl = 0   # track overall DLG status — only meaningful when calibrated

        def _cell_abs(v):
            return _cell(f"{v:.3f}" if v is not None else "—", "#dcdcdc")

        for ri, angle in enumerate(GANTRY_ANGLES):
            fe  = self._image_results[angle].get("field_edges", {})
            mlc = fe.get("mlc_total_mm")
            jaw = fe.get("jaw_total_mm")
            self._fs_angle_table.setItem(ri, 1, _vcell_dev(mlc, ref_mlc))
            self._fs_angle_table.setItem(ri, 2, _vcell_dev(jaw, ref_jaw, "jaw"))
            # Results tab: absolute + delta
            self._res_fs_angle_table.setItem(ri, 1, _cell_abs(mlc))
            self._res_fs_angle_table.setItem(ri, 2, _vcell_dev(mlc, ref_mlc))
            self._res_fs_angle_table.setItem(ri, 3, _cell_abs(jaw))
            self._res_fs_angle_table.setItem(ri, 4, _vcell_dev(jaw, ref_jaw, "jaw"))
            if is_calibrated:
                for v, ref, kind in [(mlc, ref_mlc, "mlc"), (jaw, ref_jaw, "jaw")]:
                    if v is not None:
                        worst_lvl = max(worst_lvl, _fs_level(v - ref, kind))
            if mlc is not None:
                mlc_acc.append(mlc)
                mlc_devs.append(mlc - ref_mlc)
            if jaw is not None:
                jaw_acc.append(jaw)
                jaw_devs.append(jaw - ref_jaw)

        mean_mlc = float(np.mean(mlc_acc)) if mlc_acc else None
        mean_jaw = float(np.mean(jaw_acc)) if jaw_acc else None
        self._fs_angle_table.setItem(4, 1, _vcell_dev(mean_mlc, ref_mlc))
        self._fs_angle_table.setItem(4, 2, _vcell_dev(mean_jaw, ref_jaw, "jaw"))
        self._res_fs_angle_table.setItem(4, 1, _cell_abs(mean_mlc))
        self._res_fs_angle_table.setItem(4, 2, _vcell_dev(mean_mlc, ref_mlc))
        self._res_fs_angle_table.setItem(4, 3, _cell_abs(mean_jaw))
        self._res_fs_angle_table.setItem(4, 4, _vcell_dev(mean_jaw, ref_jaw, "jaw"))

        # Maximum deviation across angles (signed, worst absolute value)
        max_mlc_dev = max(mlc_devs, key=abs) if mlc_devs else None
        max_jaw_dev = max(jaw_devs, key=abs) if jaw_devs else None

        # MLC leaf table (from G000° image) — compared to leaf baseline, not total MLC ref
        leaves = self._image_results.get(0, {}).get("mlc_leaves", [])
        leaf_devs = []
        for ri, leaf in enumerate(leaves):
            total = leaf.get("total_mm")
            self._fs_leaf_table.setItem(ri, 2, _vcell_dev(total, ref_leaf))
            # Results tab: absolute + delta
            self._res_fs_leaf_table.setItem(
                ri, 2, _cell(f"{total:.3f}" if total is not None else "—", "#dcdcdc")
            )
            self._res_fs_leaf_table.setItem(ri, 3, _vcell_dev(total, ref_leaf))
            if total is not None:
                leaf_devs.append(total - ref_leaf)
                if is_calibrated:
                    worst_lvl = max(worst_lvl, _fs_level(total - ref_leaf))

        # ── Update DLG status label (bottom bar) ─────────────────────────────
        def _dstr(v): return f"{v:+.3f}" if v is not None else "—"

        if not is_calibrated:
            dlg_text  = (f"Field Size:  Reference not set — "
                         f"go to 'Field Size QA' tab and click 'Set Current as Reference'")
            dlg_style = "font-size: 13px; color: #ffa726; font-weight: bold;"
        elif worst_lvl == 0:
            dlg_text  = (f"Field Size:  PASS   "
                         f"Max ΔMLC = {_dstr(max_mlc_dev)} mm   "
                         f"Max ΔJaw = {_dstr(max_jaw_dev)} mm")
            dlg_style = "font-size: 13px; color: #66bb6a;"
        elif worst_lvl == 1:
            dlg_text  = (f"Field Size:  WARNING   "
                         f"Max ΔMLC = {_dstr(max_mlc_dev)} mm   "
                         f"Max ΔJaw = {_dstr(max_jaw_dev)} mm")
            dlg_style = "font-size: 13px; color: #ffd600; font-weight: bold;"
        else:
            dlg_text  = (f"Field Size:  FAIL — Calibration Needed   "
                         f"Max ΔMLC = {_dstr(max_mlc_dev)} mm   "
                         f"Max ΔJaw = {_dstr(max_jaw_dev)} mm")
            dlg_style = "font-size: 13px; color: #ef5350; font-weight: bold;"
        self._dlg_status_label.setText(dlg_text)
        self._dlg_status_label.setStyleSheet(dlg_style)

        # ── Update field size PASS/FAIL banner ────────────────────────────────
        if not is_calibrated:
            self._pf_frame_fs.setStyleSheet(
                "QFrame { background-color: #2b2b2b; border-radius: 10px; }"
            )
            self._pf_label_fs.setText("FIELD  —  Reference Not Set")
            self._pf_label_fs.setStyleSheet(
                "font-size: 22px; font-weight: bold; color: #ffa726; background: transparent;"
            )
        elif worst_lvl == 0:
            self._pf_frame_fs.setStyleSheet(
                "QFrame { background-color: #1a3d1e; border-radius: 10px; }"
            )
            self._pf_label_fs.setText(
                f"FIELD  PASS    ΔMLC {_dstr(max_mlc_dev)}  ΔJaw {_dstr(max_jaw_dev)} mm"
            )
            self._pf_label_fs.setStyleSheet(
                "font-size: 22px; font-weight: bold; color: #66bb6a; background: transparent;"
            )
        elif worst_lvl == 1:
            self._pf_frame_fs.setStyleSheet(
                "QFrame { background-color: #3d3000; border-radius: 10px; }"
            )
            self._pf_label_fs.setText(
                f"FIELD  WARNING    ΔMLC {_dstr(max_mlc_dev)}  ΔJaw {_dstr(max_jaw_dev)} mm"
            )
            self._pf_label_fs.setStyleSheet(
                "font-size: 22px; font-weight: bold; color: #ffd600; background: transparent;"
            )
        else:
            self._pf_frame_fs.setStyleSheet(
                "QFrame { background-color: #3d1a1a; border-radius: 10px; }"
            )
            self._pf_label_fs.setText(
                f"FIELD  FAIL    ΔMLC {_dstr(max_mlc_dev)}  ΔJaw {_dstr(max_jaw_dev)} mm"
            )
            self._pf_label_fs.setStyleSheet(
                "font-size: 22px; font-weight: bold; color: #ef5350; background: transparent;"
            )

    # ── Event handlers ────────────────────────────────────────────────────────

    def _load_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select DICOM Directory",
            self._config.get("last_dicom_dir", str(Path.home())),
        )
        if not directory:
            return
        self._config["last_dicom_dir"] = directory
        _save_config(self._config)
        self._process_directory(directory)

    def _process_directory(self, directory, display_name=None):
        """Analyse a folder of Daily WL images (the manual import, and the staged copy of an EPID-folder session)."""
        self._dir_label.setText(display_name or Path(directory).name)
        self._pf_label.setText("Processing…")
        self._pf_label.setStyleSheet(
            "font-size: 25px; font-weight: bold; color: gray; background: transparent;"
        )
        self._pf_frame.setStyleSheet(
            "QFrame { background-color: #2b2b2b; border-radius: 10px; }"
        )
        QApplication.processEvents()

        try:
            dcm_images    = load_dicom_images(directory)
            _folder_note  = getattr(load_dicom_images, "note", "")
            if _folder_note:
                self._dir_label.setText(f"{display_name or Path(directory).name}  ({_folder_note})")
            for _ds in dcm_images.values():                    # the machine first: its log folder is used below
                _pid0 = getattr(_ds, "PatientID", None)
                if _pid0 and str(_pid0) in PATIENT_ID_MACHINE_MAP:
                    self._machine_combo.setCurrentText(PATIENT_ID_MACHINE_MAP[str(_pid0)])
                    break
            image_results = {a: analyze_image(ds) for a, ds in dcm_images.items()}
            self._import_dir = directory
            self._log_note = self._attach_logs(image_results)
            wl_results    = self._compute_results(image_results)

            self._dicom_date = None
            self._dicom_time = None
            for ds in dcm_images.values():
                for dtag, ttag in (("StudyDate", "StudyTime"),
                                   ("ContentDate", "ContentTime")):
                    raw_d = getattr(ds, dtag, None)
                    if raw_d and len(str(raw_d)) == 8:
                        try:
                            self._dicom_date = datetime.datetime.strptime(
                                str(raw_d), "%Y%m%d"
                            ).date()
                            raw_t = getattr(ds, ttag, None)
                            if raw_t:
                                ts = str(raw_t).split(".")[0].zfill(6)
                                if len(ts) >= 6:
                                    self._dicom_time = datetime.time(
                                        int(ts[0:2]), int(ts[2:4]), int(ts[4:6])
                                    )
                        except (ValueError, TypeError):
                            pass
                        break
                if self._dicom_date:
                    break

            # Fallback: parse date from folder name (never use today's date)
            if not self._dicom_date:
                import re as _re
                _fn = os.path.basename(directory.rstrip("/\\"))
                # Try 8-digit MMDDYYYY first (e.g. 06092026)
                _m = _re.search(r"(\d{2})(\d{2})(\d{4})", _fn)
                if _m:
                    try:
                        self._dicom_date = datetime.date(
                            int(_m.group(3)), int(_m.group(1)), int(_m.group(2))
                        )
                    except ValueError:
                        pass
                if not self._dicom_date:
                    # Try 6-digit MMDDYY (e.g. 061126)
                    _m = _re.search(r"(\d{2})(\d{2})(\d{2})(?!\d)", _fn)
                    if _m:
                        try:
                            self._dicom_date = datetime.date(
                                2000 + int(_m.group(3)), int(_m.group(1)), int(_m.group(2))
                            )
                        except ValueError:
                            pass

            # Auto-select machine from DICOM PatientID
            for ds in dcm_images.values():
                _pid = getattr(ds, "PatientID", None)
                if _pid and str(_pid) in PATIENT_ID_MACHINE_MAP:
                    _detected = PATIENT_ID_MACHINE_MAP[str(_pid)]
                    self._machine_combo.setCurrentText(_detected)
                    break

            self._image_results = image_results
            self._wl_results    = wl_results
            self._loaded_dir    = directory
            self._diag_fig_path = generate_diagnostic_figure(image_results, wl_results)
            self._walk_fig_path = generate_walk_circle_figure(wl_results)

            # Auto-detect picket fence image anywhere under the directory (iViewGT
            # exports nest each RTIMAGE in its own UID-named subfolder, so a
            # top-level-only scan would miss a co-located PF image)
            self._pf_results  = None
            self._pf_fig_path = None
            try:                                                   # native iView export: the picket-fence image is found from its shape
                import iview_native
                _pfs = [d for d in iview_native.scan_native(directory, APP_DIR) if d.kind == "pf"] if iview_native.is_native_folder(directory) else []
                if _pfs:
                    _pfs[-1].RTImageLabel = "PF"
                    self._pf_results  = analyze_picket_fence(_pfs[-1])
                    self._pf_fig_path = generate_pf_figure(self._pf_results)
            except Exception:
                self._pf_results = None
            for _pf_path in ([] if self._pf_results else sorted(Path(directory).rglob("*"))):
                if not _pf_path.is_file():
                    continue
                try:
                    _ds_pf = pydicom.dcmread(str(_pf_path), force=True)
                    if identify_pf_dicom(_ds_pf):
                        self._pf_results  = analyze_picket_fence(_ds_pf)
                        self._pf_fig_path = generate_pf_figure(self._pf_results)
                        break
                except Exception:
                    pass

            self._refresh_display()
            self._refresh_pf_display()
            self._report_btn.setEnabled(True)

        except Exception as exc:
            self._wl_results = self._image_results = None            # never keep the previous session's numbers under the new folder's name (they could be reported or saved as it)
            self._pf_results, self._diag_fig_path, self._walk_fig_path, self._pf_fig_path = None, None, None, None
            self._report_btn.setEnabled(False)
            QMessageBox.critical(self, "Processing Error", str(exc))
            self._pf_label.setText("— ERROR —")
            self._pf_label.setStyleSheet(
                "font-size: 25px; font-weight: bold; color: #f44336; background: transparent;"
            )

    # ── Machine logs (optional) ───────────────────────────────────────────────

    def _machine_folders(self, machine=None) -> dict:
        return (self._config.get("machine_folders") or {}).get(machine or self._machine_combo.currentText(), {})

    def _compute_results(self, image_results) -> dict:
        use = self._use_logs_cb.isChecked()
        wl_results = compute_wl_results(image_results, use_logs=use)
        wl_cent = compute_wl_results(image_results, dx_key="dx_mm_cent", dy_key="dy_mm_cent", use_logs=use)
        wl_results["walk_circle_r_cent"] = wl_cent["walk_circle_r"]
        wl_results["pass_fail_cent"]     = wl_cent["pass_fail"]
        return wl_results

    def _attach_logs(self, image_results) -> str:
        """Pair each image with its machine log (optional). Returns a one-line note; never raises: no folder, no logs, or no match all mean 'analyse as before'."""
        for ir in image_results.values():
            ir.pop("log", None)
        folders = [f for f in (self._machine_folders().get("logs", ""), getattr(self, "_import_dir", "")) if f]
        if not folders:
            return "no machine-log folder set"
        try:
            import wl_sources
            imgs = {a: (ir["when"], ir["gantry"]) for a, ir in image_results.items() if ir.get("when")}
            got, seen = {}, 0
            for folder in folders:                                   # the machine's log folder, then any logs delivered with the images
                idx = wl_sources.LogIndex(folder, APP_DIR)
                seen += len(idx.records)
                got = idx.match(imgs) if idx.records else {}
                if got:
                    break
            if not seen:
                return "no log files found in the log folder"
            n = 0
            for a, rec in got.items():
                info = wl_sources.aperture_shift(rec)
                if info:
                    image_results[a]["log"] = info
                    n += 1
            return f"machine log matched for {n} of {len(image_results)} images" if n else "no log matched these images"
        except Exception as exc:                                  # a log problem must never stop the analysis
            return f"machine logs not used ({exc})"

    def _logs_toggled(self, on):
        self._config["use_machine_logs"] = bool(on)
        _save_config(self._config)
        if getattr(self, "_image_results", None):
            self._wl_results = self._compute_results(self._image_results)
            self._diag_fig_path = generate_diagnostic_figure(self._image_results, self._wl_results)
            self._walk_fig_path = generate_walk_circle_figure(self._wl_results)
            self._refresh_display()

    # ── EPID folder ───────────────────────────────────────────────────────────

    def _save_epid_mark(self, sessions):
        """Remember how far the EPID folder has been dealt with, so the next check reads only what is newer."""
        import wl_sources
        folder, newest = getattr(self, "_epid_scanned", (None, None))
        if folder and newest:
            wl_sources.save_scan_mark(folder, APP_DIR, sessions, newest)

    def _check_epid_folder(self, full=False):
        import wl_sources
        machine = self._machine_combo.currentText()
        folder = self._machine_folders(machine).get("epid", "")
        if not folder:
            if QMessageBox.question(self, "EPID folder not set",
                                    f"No EPID image folder is set for {machine}.\n\nOpen Settings > Data Folders to set it?") == QMessageBox.StandardButton.Yes:
                self._show_settings(tab="folders")
            return
        conn = sqlite3.connect(DB_PATH); wl_sources.init_registry(conn); n_done = len(wl_sources.processed_sops(conn, machine)); conn.close()
        since = None if (full is True or not n_done) else wl_sources.scan_mark(folder, APP_DIR)      # nothing imported yet, or a full check asked for: read everything
        prog = QProgressDialog("Reading the EPID folder..." if since is None else "Looking for images newer than the last check...", "Cancel", 0, 100, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)
        cancelled = []

        def _p(i, n, text=None):
            if text:
                prog.setLabelText(text)
            prog.setMaximum(max(n, 1)); prog.setValue(min(i, n))
            QApplication.processEvents()
            if prog.wasCanceled():
                cancelled.append(1)
                raise InterruptedError()
        try:
            sessions = wl_sources.scan_epid_folder(folder, APP_DIR, since=since, progress=lambda i, n: _p(i, n, f"Reading image headers... {i} of {n}"))
            self._epid_scanned = (folder, getattr(wl_sources.scan_epid_folder, "newest", None))
        except InterruptedError:
            prog.close(); return
        except Exception as exc:
            prog.close(); QMessageBox.warning(self, "EPID folder", str(exc)); return
        conn = sqlite3.connect(DB_PATH)
        wl_sources.init_registry(conn)
        done = wl_sources.processed_sops(conn)
        conn.close()
        sessions = [x for x in sessions if PATIENT_ID_MACHINE_MAP.get(x.patient, machine) == machine]
        for x in sessions:
            x.machine = machine
        wl_sources.classify(sessions, done)
        self._save_epid_mark(sessions)
        new = [x for x in sessions if x.status == "new"]
        # quick result for the newest complete new sessions, so the list shows what each one would give
        todo = [x for x in new if x.complete][:40]
        try:
            for k, x in enumerate(todo):
                _p(k, len(todo), f"Analysing new session {k + 1} of {len(todo)} ({x.when:%Y-%m-%d %H:%M})...")
                try:
                    ir = {a: analyze_image(read_image(path)) for a, path in x.files.items()}
                    self._attach_logs_for(ir, machine)
                    r = compute_wl_results(ir, use_logs=self._use_logs_cb.isChecked())
                    mlc = [v["field_edges"].get("mlc_total_mm") for v in ir.values() if v.get("field_edges", {}).get("mlc_total_mm") is not None]
                    jaw = [v["field_edges"].get("jaw_total_mm") for v in ir.values() if v.get("field_edges", {}).get("jaw_total_mm") is not None]
                    pfm = None
                    if x.pf:
                        try:
                            pfm = float(analyze_picket_fence(read_image(x.pf))["max_deviation_mm"])
                        except Exception:
                            pfm = None
                    x.preview = {"displacement": r["max_2d_walk_mm"], "pass": r["pass_fail"], "logs": r["logs_used"],
                                 "mlc": float(np.mean(mlc)) if mlc else None, "jaw": float(np.mean(jaw)) if jaw else None, "pf_max": pfm}
                    for v in ir.values():
                        v.pop("pixel_array", None)
                except Exception:
                    x.preview = None
        except InterruptedError:
            pass
        prog.close()
        if not new:
            QMessageBox.information(self, "EPID folder",
                                    f"No new Daily WL sessions in\n{folder}\n\n" + ("(Only images newer than the last check were read; use Full Rescan to read the whole folder.)" if since is not None
                                                                                  else f"({len(sessions)} session(s) found, all already processed or set aside.)"))
            return
        if len(new) == 1 and new[0].complete:
            self._open_epid_session(new[0])
            return
        EpidSessionsDialog(self, sessions, machine).exec()

    def _import_all_sessions(self, sessions, machine):
        """Analyse and save many sessions (see daily_batch): window stays responsive, can be cancelled, one bad session never stops the run."""
        import daily_batch
        prog = QProgressDialog("Importing sessions...", "Cancel", 0, max(len(sessions), 1), self)
        prog.setWindowModality(Qt.WindowModality.WindowModal); prog.setMinimumDuration(0)

        def _p(i, n, text=""):
            prog.setMaximum(max(n, 1)); prog.setValue(min(i, n))
            if text:
                prog.setLabelText(text)
            QApplication.processEvents()
        folders = [self._machine_folders(machine).get("logs", "")]
        try:
            out = daily_batch.import_sessions(sys.modules[__name__], sessions, machine, self._physicist_combo.currentText(), log_folders=folders, cache_dir=APP_DIR,
                                              use_logs=self._use_logs_cb.isChecked(), progress=_p, should_cancel=prog.wasCanceled)
        except Exception as exc:                                    # nothing in an import may take the application down
            prog.close(); QMessageBox.warning(self, "Import", f"The import stopped: {type(exc).__name__}: {exc}\nSessions saved so far are kept; run it again to continue."); return
        prog.close()
        text = daily_batch.summary_text(out)
        try:
            (APP_DIR / "daily_import.log").write_text(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {machine}\n{text}\n")
        except OSError:
            pass
        box = QMessageBox(self); box.setWindowTitle("Import finished"); box.setText(text.split("\n", 1)[0]); box.setDetailedText(text); box.exec()

    def _attach_logs_for(self, image_results, machine):
        keep = self._machine_combo.currentText()
        folder = self._machine_folders(machine).get("logs", "")
        if not folder:
            return
        try:
            import wl_sources
            idx = wl_sources.LogIndex(folder, APP_DIR)
            got = idx.match({a: (ir["when"], ir["gantry"]) for a, ir in image_results.items() if ir.get("when")})
            for a, rec in got.items():
                info = wl_sources.aperture_shift(rec)
                if info:
                    image_results[a]["log"] = info
        except Exception:
            pass

    def _find_session_images(self, machine, when, tol_min=10):
        """The EPID-folder session nearest to `when` (same day, within `tol_min` minutes), or None; used to reopen a stored result."""
        import wl_sources
        folder = self._machine_folders(machine).get("epid", "")
        if not folder or not Path(folder).is_dir():
            return None
        try:
            sessions = wl_sources.scan_epid_folder(folder, APP_DIR, since=when.timestamp() - 86400)
        except Exception:
            return None
        cand = [x for x in sessions if x.complete and x.when.date() == when.date() and abs((x.when - when).total_seconds()) <= tol_min * 60
                and PATIENT_ID_MACHINE_MAP.get(x.patient, machine) == machine]
        if not cand:
            return None
        best = min(cand, key=lambda x: abs((x.when - when).total_seconds()))
        best.machine = machine
        return best

    def _open_epid_session(self, sess, queue=None, sessions=None):
        import wl_sources
        if queue is not None:
            self._queue, self._queue_all = list(queue), sessions
        elif not getattr(self, "_queue", None) or sess not in self._queue:
            self._queue, self._queue_all = [], None
        self._queue_cur = sess
        if sess.machine:
            self._machine_combo.setCurrentText(sess.machine)
        old = getattr(self, "_staged_dir", None)
        self._staged_dir = str(wl_sources.stage_session(sess))
        self._process_directory(self._staged_dir, f"EPID: {sess.when:%Y-%m-%d %H:%M}  ({sess.machine})")
        if old:
            shutil.rmtree(old, ignore_errors=True)
        self._update_queue_bar()

    def _update_queue_bar(self):
        q, cur = getattr(self, "_queue", []), getattr(self, "_queue_cur", None)
        if not q or cur not in q:
            self._queue_bar.setVisible(False); return
        i = q.index(cur)
        self._queue_label.setText(f"Reviewing new session {i + 1} of {len(q)}:  {cur.when:%a %Y-%m-%d %H:%M}"
                                  + ("" if cur.status == "new" else f"   ({cur.status})"))
        self._queue_prev.setEnabled(i > 0); self._queue_next.setEnabled(i < len(q) - 1)
        self._queue_bar.setVisible(True)

    def _queue_step(self, n):
        q, cur = getattr(self, "_queue", []), getattr(self, "_queue_cur", None)
        if q and cur in q and 0 <= q.index(cur) + n < len(q):
            self._open_epid_session(q[q.index(cur) + n])

    def _queue_list(self):
        """Back to the grouped list of sessions found in the EPID folder."""
        sessions = getattr(self, "_queue_all", None)
        if not sessions:
            return
        import wl_sources
        conn = sqlite3.connect(DB_PATH); done = wl_sources.processed_sops(conn); conn.close()
        wl_sources.classify(sessions, done)
        self._queue = []; self._update_queue_bar()
        EpidSessionsDialog(self, sessions, self._machine_combo.currentText()).exec()

    def _refresh_pf_display(self):
        """Update PF banner, stats card, and figure tab from self._pf_results."""
        if self._pf_results is None:
            self._pf_frame_picket.setVisible(False)
            self._pf_stats_card.setVisible(False)
            self._pf_img_label.setText(
                "Load a Picket Fence DICOM directory or file to view results.\n"
                "PF images in the same WL directory are detected automatically."
            )
            self._pf_img_label.setPixmap(QPixmap())
            return

        pf    = self._pf_results
        ok    = pf["pass_fail"]
        maxd  = pf["max_deviation_mm"]
        rms   = pf["rms_deviation_mm"]
        n     = pf["n_leaves"]
        avg_w = pf_avg_width_central_mm(pf)

        # Banner
        self._pf_frame_picket.setVisible(True)
        if ok:
            self._pf_frame_picket.setStyleSheet(
                "QFrame { background-color: #1a3d1e; border-radius: 10px; }"
            )
            self._pf_label_picket.setStyleSheet(
                "font-size: 22px; font-weight: bold; color: #66bb6a; background: transparent;"
            )
            self._pf_label_picket.setText(
                f"✓  PF PASS  —  Max |Δ| = {maxd:.2f} mm  (Tol ≤ {PF_TOLERANCE_MM:.1f} mm)"
            )
        else:
            self._pf_frame_picket.setStyleSheet(
                "QFrame { background-color: #3d1a1a; border-radius: 10px; }"
            )
            self._pf_label_picket.setStyleSheet(
                "font-size: 22px; font-weight: bold; color: #ef5350; background: transparent;"
            )
            self._pf_label_picket.setText(
                f"✗  PF FAIL  —  Max |Δ| = {maxd:.2f} mm  (Tol ≤ {PF_TOLERANCE_MM:.1f} mm)"
            )

        # Stats card
        self._pf_stats_card.setVisible(True)
        self._pf_stat_leaves.setText(str(n))
        col_max = "#4caf50" if maxd < PF_WARN_MM else \
                  "#ff9800" if maxd < PF_TOLERANCE_MM else "#ef5350"
        self._pf_stat_max.setText(f"{maxd:.3f}")
        self._pf_stat_max.setStyleSheet(
            f"color:{col_max}; font-size:17px; font-weight:bold; "
            "font-family:monospace; background:transparent;"
        )
        self._pf_stat_rms.setText(f"{rms:.3f}")
        self._pf_stat_width.setText(f"{avg_w:.2f}")

        # Figure
        if self._pf_fig_path:
            try:
                import io
                from PIL import Image as PILImage
                pil_img = PILImage.open(self._pf_fig_path)
                avail_w = max(900, self.width() - 60)
                scale_h = int(pil_img.height * avail_w / pil_img.width)
                pil_img = pil_img.resize((avail_w, scale_h), PILImage.LANCZOS)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                pixmap = QPixmap()
                pixmap.loadFromData(buf.getvalue())
                self._pf_img_label.setPixmap(pixmap)
                self._pf_img_label.adjustSize()
            except Exception as exc:
                self._pf_img_label.setText(f"Figure error: {exc}")

    def _refresh_display(self):
        if self._wl_results is None:
            return

        wl   = self._wl_results
        walk = wl["max_2d_walk_mm"]
        ok   = wl["pass_fail"]

        if ok:
            self._pf_frame.setStyleSheet(
                "QFrame { background-color: #1a3d1e; border-radius: 10px; }"
            )
            self._pf_label.setText(f"WL  PASS    {walk:.3f} mm")
            self._pf_label.setStyleSheet(
                "font-size: 25px; font-weight: bold; color: #66bb6a; background: transparent;"
            )
        else:
            self._pf_frame.setStyleSheet(
                "QFrame { background-color: #3d1a1a; border-radius: 10px; }"
            )
            self._pf_label.setText(f"WL  FAIL    {walk:.3f} mm")
            self._pf_label.setStyleSheet(
                "font-size: 25px; font-weight: bold; color: #ef5350; background: transparent;"
            )

        walk_cent = wl.get("walk_circle_r_cent")
        if walk_cent is not None:
            self._walk_label.setText(
                f"Max displacement:  Edge = {walk:.3f} mm  │  Centroid = {walk_cent:.3f} mm"
                f"   (Tolerance ≤ {TOLERANCE_MM:.1f} mm)"
            )
        else:
            self._walk_label.setText(
                f"Max displacement: {walk:.3f} mm   "
                f"(Tolerance ≤ {TOLERANCE_MM:.1f} mm)"
            )
        sx = wl["setup_x"]; sy = wl["setup_y"]; sz = wl["setup_z"]
        _logtxt = (f"   │   machine log applied to {wl['logs_used']} of 4 images (largest field-centre shift {wl['log_shift_max']:.2f} mm)"
                   if wl.get("logs_used") else f"   │   machine log: {getattr(self, '_log_note', 'not used')}")
        _weak = [a for a, r in (getattr(self, "_image_results", None) or {}).items() if not (r.get("void_fit") or {}).get("ok", True)]
        _voidtxt = (f"   │   ⚠ void not found reliably at G{', G'.join(str(a) for a in sorted(_weak))} - check the phantom and the images"
                    if _weak else "")
        self._corr_sub.setText(
            f"CBCT setup: Lat={sx:+.3f} mm   SI={sy:+.3f} mm   AP={sz:+.3f} mm" + _logtxt + _voidtxt
        )
        self._corr_sub.setStyleSheet(f"font-size: 12px; color: {'#ffb74d' if _weak else '#90caf9'}; background: transparent;")

        def _color(v):
            if v < TOLERANCE_MM / 2:
                return "#66bb6a"
            if v < TOLERANCE_MM:
                return "#ffa726"
            return "#ef5350"

        def _set(key, text, color):
            lbl = self._table_labels[key]
            lbl.setText(text)
            lbl.setStyleSheet(
                f"font-size: 17px; font-family: monospace; color: {color};"
            )

        for angle in GANTRY_ANGLES:
            r = wl["per_angle"][angle]
            corr_dr = np.sqrt(r["rel_dx"] ** 2 + r["rel_dy"] ** 2)
            _set(f"corr_{angle}_dx", f"{r['rel_dx']:+.3f}", _color(abs(r["rel_dx"])))
            _set(f"corr_{angle}_dy", f"{r['rel_dy']:+.3f}", _color(abs(r["rel_dy"])))
            _set(f"corr_{angle}_dr", f"{corr_dr:.3f}",      _color(corr_dr))

        self._refresh_field_size_display()
        self._update_walk_fig()
        self._update_diag_image()

    def _update_walk_fig(self):
        if not self._walk_fig_path or not os.path.exists(self._walk_fig_path):
            return
        try:
            import io
            from PIL import Image as PILImage
            pil_img = PILImage.open(self._walk_fig_path)
            # Fit inside the label's current width; keep square-ish aspect
            avail_w = max(300, self._walk_fig_label.width())
            scale_h = int(pil_img.height * avail_w / pil_img.width)
            pil_img = pil_img.resize((avail_w, scale_h), PILImage.LANCZOS)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            self._walk_fig_label.setPixmap(pixmap)
        except Exception:
            pass

    def _update_diag_image(self):
        if not self._diag_fig_path or not os.path.exists(self._diag_fig_path):
            return
        try:
            import io
            from PIL import Image as PILImage
            pil_img = PILImage.open(self._diag_fig_path)
            avail_w = max(900, self.width() - 60)
            scale_h = int(pil_img.height * avail_w / pil_img.width)
            pil_img = pil_img.resize((avail_w, scale_h), PILImage.LANCZOS)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            self._diag_img_label.setPixmap(pixmap)
            self._diag_img_label.adjustSize()
        except Exception as exc:
            self._diag_img_label.setText(f"Image display error:\n{exc}")
            self._diag_img_label.setPixmap(QPixmap())

    def _generate_report(self):
        if self._wl_results is None:
            QMessageBox.warning(self, "No Data", "Load DICOM images first.")
            return

        machine   = self._machine_combo.currentText()
        physicist = self._physicist_combo.currentText()

        _fn_date = self._dicom_date.strftime("%Y%m%d") if self._dicom_date else "UNKNOWN"
        default_name = f"WL_QA_{machine.replace(' ', '_')}_{_fn_date}.pdf"
        last_dirs = self._config.setdefault("last_report_dir_by_machine", {})
        last_dir = last_dirs.get(machine, self._config.get("last_report_dir", str(Path.home())))
        initial_path = str(Path(last_dir) / default_name)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Daily QA Report",
            initial_path,
            "PDF files (*.pdf)",
        )
        if not save_path:
            return
        last_dirs[machine] = str(Path(save_path).parent)
        self._config["last_report_dir"] = str(Path(save_path).parent)
        _save_config(self._config)

        try:
            generate_pdf_report(
                self._wl_results,
                self._image_results,
                save_path,
                diag_fig_path=self._diag_fig_path,
                walk_fig_path=self._walk_fig_path,
                machine_name=machine,
                physicist_name=physicist,
                dicom_date=self._dicom_date,
                dicom_time=self._dicom_time,
                pf_results=self._pf_results,
                pf_fig_path=self._pf_fig_path,
            )
            _when = (datetime.datetime.combine(self._dicom_date, self._dicom_time or datetime.time(0, 0)) if self._dicom_date else None)
            _save_to_db(self._wl_results, self._image_results, machine, physicist, when=_when, pf_results=self._pf_results)
            QMessageBox.information(self, "Report Saved", f"Report saved to:\n{save_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Report Error", str(exc)); return
        cur, q = getattr(self, "_queue_cur", None), getattr(self, "_queue", [])
        if q and cur in q:
            cur.status = "processed"
            rest = [y for y in q[q.index(cur) + 1:] if y.status == "new"]
            if rest and QMessageBox.question(self, "Next session", f"{len(rest)} more new session(s) to review.\n\nOpen the next one ({rest[0].when:%a %Y-%m-%d %H:%M})?") == QMessageBox.StandardButton.Yes:
                self._open_epid_session(rest[0])
            else:
                self._update_queue_bar()

    # ── Settings dialog ──────────────────────────────────────────────────────

    def _build_settings_folders_tab(self) -> QWidget:
        """Per machine: the EPID image folder 'Check EPID Folder' searches, and the (optional) machine-log folder."""
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.addWidget(QLabel("Where each machine's EPID images arrive (searched by 'Check EPID Folder') and, optionally, its machine log files.\n"
                               "The log folder can stay empty: the analysis then simply uses the images as measured."))
        for machine in self._machines:
            box = QGroupBox(machine)
            form = QFormLayout(box)
            cur = self._machine_folders(machine)
            for key, label in (("epid", "EPID image folder"), ("logs", "Machine log folder (optional)")):
                row = QHBoxLayout()
                edit = QLineEdit(cur.get(key, ""))
                edit.setPlaceholderText("folder path (local or network share)")
                btn = QPushButton("Browse...")

                def _save(_=None, m=machine, k=key, e=edit):
                    self._config.setdefault("machine_folders", {}).setdefault(m, {})[k] = e.text().strip()
                    _save_config(self._config)

                def _browse(_=False, e=edit, sv=_save):
                    d = QFileDialog.getExistingDirectory(tab, "Choose folder", e.text() or str(Path.home()))
                    if d:
                        e.setText(d); sv()
                edit.editingFinished.connect(_save)
                btn.clicked.connect(_browse)
                row.addWidget(edit, 1); row.addWidget(btn)
                form.addRow(label, row)
            outer.addWidget(box)
        outer.addStretch()
        return tab

    def _show_settings(self, tab=None):
        """Open the Settings dialog: machines, physicists, data folders, manual field-size reference."""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{APP_NAME} — Settings")
        dlg.resize(560, 480)
        dlg_layout = QVBoxLayout(dlg)

        tabs = QTabWidget()
        dlg_layout.addWidget(tabs)

        tabs.addTab(self._build_settings_list_tab("machines"), "Machines")
        tabs.addTab(self._build_settings_list_tab("physicists"), "Physicists")
        tabs.addTab(self._build_settings_folders_tab(), "Data Folders")
        tabs.addTab(self._build_settings_refs_tab(), "Field Size Reference")
        if tab == "folders":
            tabs.setCurrentIndex(2)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        dlg_layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

        dlg.exec()

    def _build_settings_list_tab(self, kind: str) -> QWidget:
        """kind is 'machines' or 'physicists' — a simple add/remove name list."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        source = self._machines if kind == "machines" else self._physicists
        singular = "machine" if kind == "machines" else "physicist"

        list_widget = QListWidget()
        list_widget.addItems(source)
        layout.addWidget(list_widget)

        btn_row = QHBoxLayout()
        add_btn = QPushButton(f"Add {singular.title()}")
        remove_btn = QPushButton(f"Remove Selected")
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        def _add():
            text, ok = QInputDialog.getText(
                tab, f"Add {singular.title()}", f"{singular.title()} name:"
            )
            text = text.strip()
            if not ok or not text:
                return
            if text in source:
                QMessageBox.warning(tab, "Duplicate", f"'{text}' is already in the list.")
                return
            source.append(text)
            list_widget.addItem(text)
            self._save_machines_physicists()

        def _remove():
            item = list_widget.currentItem()
            if item is None:
                return
            name = item.text()
            if len(source) <= 1:
                QMessageBox.warning(
                    tab, "Cannot Remove", f"At least one {singular} must remain."
                )
                return
            reply = QMessageBox.question(
                tab, f"Remove {singular.title()}",
                f"Remove '{name}' from the {singular} list?\n\n"
                "Historical records already saved to the trend database are not affected.",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            source.remove(name)
            list_widget.takeItem(list_widget.row(item))
            self._save_machines_physicists()

        add_btn.clicked.connect(_add)
        remove_btn.clicked.connect(_remove)
        return tab

    def _save_machines_physicists(self):
        """Persist the current machine/physicist lists and refresh the combo boxes."""
        self._config["machines"]   = list(self._machines)
        self._config["physicists"] = list(self._physicists)
        _save_config(self._config)

        cur_machine = self._machine_combo.currentText()
        self._machine_combo.blockSignals(True)
        self._machine_combo.clear()
        self._machine_combo.addItems(self._machines)
        if cur_machine in self._machines:
            self._machine_combo.setCurrentText(cur_machine)
        self._machine_combo.blockSignals(False)
        self._update_field_ref_labels()

        cur_physicist = self._physicist_combo.currentText()
        self._physicist_combo.clear()
        self._physicist_combo.addItems(self._physicists)
        if cur_physicist in self._physicists:
            self._physicist_combo.setCurrentText(cur_physicist)

    def _build_settings_refs_tab(self) -> QWidget:
        """Manual numeric entry of the MLC/Jaw field-size reference per machine."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        note = QLabel(
            "Manually enter this machine's reference (baseline) field size, e.g. after a "
            "physical calibration where the exact values are already known. This is an "
            "alternative to 'Set Current as Reference' on the Field Size QA tab."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        layout.addWidget(note)

        form = QHBoxLayout()
        form.addWidget(QLabel("Machine:"))
        machine_combo = QComboBox()
        machine_combo.addItems(self._machines)
        machine_combo.setCurrentText(self._machine_combo.currentText())
        form.addWidget(machine_combo)
        form.addStretch()
        layout.addLayout(form)

        grid = QGridLayout()
        grid.addWidget(QLabel("MLC reference (mm):"), 0, 0)
        mlc_spin = QDoubleSpinBox()
        mlc_spin.setRange(0.0, 200.0)
        mlc_spin.setDecimals(3)
        mlc_spin.setSingleStep(0.1)
        grid.addWidget(mlc_spin, 0, 1)

        grid.addWidget(QLabel("Jaw reference (mm):"), 1, 0)
        jaw_spin = QDoubleSpinBox()
        jaw_spin.setRange(0.0, 200.0)
        jaw_spin.setDecimals(3)
        jaw_spin.setSingleStep(0.1)
        grid.addWidget(jaw_spin, 1, 1)
        layout.addLayout(grid)

        def _load_for_machine():
            refs = self._get_field_refs(machine_combo.currentText())
            mlc_spin.setValue(refs["mlc"])
            jaw_spin.setValue(refs["jaw"])

        _load_for_machine()
        machine_combo.currentIndexChanged.connect(_load_for_machine)

        save_btn = QPushButton("Save Reference")

        def _save():
            machine = machine_combo.currentText()
            existing = self._config.get("field_refs", {}).get(machine, {})
            new_ref = dict(existing)
            new_ref["mlc"] = mlc_spin.value()
            new_ref["jaw"] = jaw_spin.value()
            new_ref.setdefault("leaf", mlc_spin.value())
            if "field_refs" not in self._config:
                self._config["field_refs"] = {}
            self._config["field_refs"][machine] = new_ref
            _save_config(self._config)
            self._update_field_ref_labels()
            self._refresh_field_size_display()
            QMessageBox.information(
                tab, "Reference Saved",
                f"Field size reference for {machine} updated:\n"
                f"MLC={new_ref['mlc']:.3f} mm,  Jaw={new_ref['jaw']:.3f} mm"
            )

        save_btn.clicked.connect(_save)
        layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()
        return tab

    # ── Trend analysis (machine → test picker) ──────────────────────────────

    import daily_trends as _trends
    _TREND_TESTS = list(_trends.KEYS)

    def _trend_machines_with_data(self) -> list:
        """Distinct machine names present in the trend DB (Winston-Lutz or picket fence), preferring configured order."""
        import daily_trends
        db_machines = set(daily_trends.machines_with_data(DB_PATH))
        ordered = [m for m in self._machines if m in db_machines]
        ordered += sorted(db_machines - set(ordered))
        return ordered

    def _show_trends(self):
        """Open the trend dialog: pick a machine, then an individual test to plot."""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{APP_NAME} — Trend Analysis")
        dlg.resize(1050, 720)
        dlg_layout = QVBoxLayout(dlg)

        machines = self._trend_machines_with_data()
        if not machines:
            lbl = QLabel(
                "No records in database yet.\n"
                "Generate a report to save the first record."
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 15px;")
            dlg_layout.addWidget(lbl)
            dlg.exec()
            return

        stack = QStackedWidget()
        dlg_layout.addWidget(stack, stretch=1)

        state = {"machine": None, "test": None}

        machine_page = self._build_trend_machine_page(machines, stack, state)
        stack.addWidget(machine_page)
        stack.setCurrentWidget(machine_page)

        dlg.exec()

    def _build_trend_machine_page(self, machines: list, stack: QStackedWidget, state: dict) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Select a machine")
        title.setStyleSheet("font-size: 17px; font-weight: bold;")
        layout.addWidget(title)

        list_widget = QListWidget()
        list_widget.addItems(machines)
        layout.addWidget(list_widget, stretch=1)

        next_btn = QPushButton("Next  →")
        next_btn.setEnabled(False)
        layout.addWidget(next_btn, alignment=Qt.AlignmentFlag.AlignRight)

        list_widget.currentRowChanged.connect(lambda r: next_btn.setEnabled(r >= 0))
        list_widget.itemDoubleClicked.connect(
            lambda item: self._trend_go_to_test_page(item.text(), stack, state)
        )
        next_btn.clicked.connect(
            lambda: self._trend_go_to_test_page(list_widget.currentItem().text(), stack, state)
        )
        return page

    def _trend_go_to_test_page(self, machine: str, stack: QStackedWidget, state: dict):
        state["machine"] = machine
        old = state.get("test_page")
        page = self._build_trend_test_page(machine, stack, state)
        stack.addWidget(page)
        stack.setCurrentWidget(page)
        state["test_page"] = page
        if old is not None:
            stack.removeWidget(old)
            old.deleteLater()

    def _build_trend_test_page(self, machine: str, stack: QStackedWidget, state: dict) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        header = QHBoxLayout()
        back_btn = QPushButton("←  Change Machine")
        back_btn.clicked.connect(lambda: stack.setCurrentIndex(0))
        header.addWidget(back_btn)
        header.addStretch()
        title = QLabel(f"{machine} — select a test")
        title.setStyleSheet("font-size: 17px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        list_widget = QListWidget()
        list_widget.addItems([label for _, label in self._TREND_TESTS])
        layout.addWidget(list_widget, stretch=1)

        next_btn = QPushButton("View Trend  →")
        next_btn.setEnabled(False)
        layout.addWidget(next_btn, alignment=Qt.AlignmentFlag.AlignRight)

        def _go(row: int):
            if row < 0:
                return
            key = self._TREND_TESTS[row][0]
            state["test"] = key
            old = state.get("chart_page")
            chart_page = self._build_trend_chart_page(machine, key, stack, state)
            stack.addWidget(chart_page)
            stack.setCurrentWidget(chart_page)
            state["chart_page"] = chart_page
            if old is not None:
                stack.removeWidget(old)
                old.deleteLater()

        list_widget.currentRowChanged.connect(lambda r: next_btn.setEnabled(r >= 0))
        list_widget.itemDoubleClicked.connect(lambda item: _go(list_widget.currentRow()))
        next_btn.clicked.connect(lambda: _go(list_widget.currentRow()))
        return page

    def _build_trend_chart_page(self, machine: str, test_key: str, stack: QStackedWidget, state: dict) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        header = QHBoxLayout()
        back_btn = QPushButton("←  Change Test")
        back_btn.clicked.connect(lambda: stack.setCurrentWidget(state["test_page"]))
        header.addWidget(back_btn)
        change_machine_btn = QPushButton("Change Machine")
        change_machine_btn.clicked.connect(lambda: stack.setCurrentIndex(0))
        header.addWidget(change_machine_btn)
        header.addStretch()
        layout.addLayout(header)

        test_label = dict(self._TREND_TESTS)[test_key]
        import daily_trends
        refs = self._get_field_refs(machine)
        limits = {"walk": TOLERANCE_MM, "mlc": (FIELD_SIZE_WARN_MM, FIELD_SIZE_FAIL_MM), "jaw": (FIELD_SIZE_JAW_WARN_MM, FIELD_SIZE_JAW_FAIL_MM), "pf": (PF_WARN_MM, PF_TOLERANCE_MM)}
        try:
            res = daily_trends.build(DB_PATH, machine, test_key, refs, limits)
        except Exception as exc:                                    # a trend problem must never take the application down
            res = {"png": None, "summary": f"The trend could not be drawn: {type(exc).__name__}: {exc}", "headers": [], "rows": []}

        note = QLabel(res["summary"])
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 13px; padding: 4px;")
        layout.addWidget(note)
        if not res["png"]:
            layout.addStretch(1)
            return page

        pixmap = QPixmap(); pixmap.loadFromData(res["png"])
        chart_label = QLabel()
        chart_label.setPixmap(pixmap.scaledToWidth(min(pixmap.width(), 1000), Qt.TransformationMode.SmoothTransformation))
        chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll = QScrollArea(); scroll.setWidget(chart_label); scroll.setWidgetResizable(True); scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        layout.addWidget(scroll, stretch=1)

        hdrs = res["headers"]
        table = QTableWidget(len(res["rows"]), len(hdrs))
        table.setHorizontalHeaderLabels(hdrs)
        table.setMaximumHeight(190)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        for ri, row in enumerate(reversed(res["rows"])):                 # newest first
            for ci, val in enumerate(row[:-1]):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if ci == len(row) - 2:
                    item.setForeground(QBrush(QColor(row[-1])))
                table.setItem(ri, ci, item)
        layout.addWidget(table)
        return page


# ── Database ──────────────────────────────────────────────────────────────────

def _init_db():
    """Create the wl_records table if it does not already exist, and migrate schema."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wl_records (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            time            TEXT NOT NULL,
            machine_name    TEXT NOT NULL,
            physicist_name  TEXT NOT NULL,
            walk_circle_r   REAL NOT NULL,
            pass_fail       INTEGER NOT NULL,
            baseline_dx     REAL,
            baseline_dy     REAL,
            baseline_dz     REAL,
            raw_dx_G0       REAL, raw_dy_G0   REAL,
            raw_dx_G90      REAL, raw_dy_G90  REAL,
            raw_dx_G180     REAL, raw_dy_G180 REAL,
            raw_dx_G270     REAL, raw_dy_G270 REAL
        )
    """)
    # Migrate older databases incrementally
    for col_def in [
        "ALTER TABLE wl_records ADD COLUMN baseline_dz REAL",
        "ALTER TABLE wl_records ADD COLUMN avg_mlc_mm REAL",
        "ALTER TABLE wl_records ADD COLUMN avg_jaw_mm REAL",
        "ALTER TABLE wl_records ADD COLUMN logs_used INTEGER",
    ]:
        try:
            conn.execute(col_def)
        except sqlite3.OperationalError:
            pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pf_records (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            time            TEXT NOT NULL,
            machine_name    TEXT NOT NULL,
            physicist_name  TEXT,
            max_dev_mm      REAL NOT NULL,       -- largest |leaf deviation|
            rms_dev_mm      REAL NOT NULL,
            n_leaves        INTEGER,
            worst_leaf      INTEGER,             -- index (0 = most negative y) of the leaf with the largest |deviation|
            worst_dev_mm    REAL,                -- its signed deviation
            avg_width_mm    REAL,
            pass_fail       INTEGER,
            leaf_y_json     TEXT,                -- leaf positions along y (mm)
            leaf_dev_json   TEXT                 -- the signed deviation of every leaf (mm)
        )
    """)
    import wl_sources
    wl_sources.init_registry(conn)
    conn.commit()
    conn.close()


def _save_pf(conn, pf_results: dict, machine: str, physicist: str, when: "datetime.datetime | None" = None) -> None:
    """One picket-fence record (the per-leaf deviations are kept, so the trend can show whether the same leaf is repeatedly the worst). A record for the same machine and time is replaced."""
    import json
    d = np.asarray(pf_results["leaf_deviations_mm"], float)
    if not d.size:
        return
    sd, st_ = pf_results.get("study_date"), pf_results.get("study_time")
    if sd is not None:
        t0 = st_ if isinstance(st_, datetime.time) else datetime.time(0, 0)
        now = datetime.datetime.combine(sd, t0)
    else:
        now = when or datetime.datetime.now()
    k = int(np.argmax(np.abs(d)))
    try:
        avg_w = float(pf_avg_width_central_mm(pf_results))
    except Exception:
        avg_w = None
    conn.execute("DELETE FROM pf_records WHERE machine_name=? AND date=? AND time=?", (machine, now.strftime("%Y-%m-%d"), now.strftime("%H:%M")))
    conn.execute(
        "INSERT INTO pf_records (date, time, machine_name, physicist_name, max_dev_mm, rms_dev_mm, n_leaves, worst_leaf, worst_dev_mm, avg_width_mm, pass_fail, leaf_y_json, leaf_dev_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (now.strftime("%Y-%m-%d"), now.strftime("%H:%M"), machine, physicist, float(np.max(np.abs(d))), float(np.sqrt(np.mean(d ** 2))), int(d.size), k, float(d[k]), avg_w,
         1 if pf_results.get("pass_fail") else 0, json.dumps([round(float(v), 2) for v in np.asarray(pf_results.get("leaf_y_mm", []), float)]),
         json.dumps([round(float(v), 4) for v in d])))


def _save_to_db(wl_results: dict, image_results: dict, machine: str, physicist: str, when: "datetime.datetime | None" = None, pf_results: "dict | None" = None):
    """Insert one QA session record into the database. `when` = when the images were ACQUIRED (the record's date and time, so trends follow the measurement date,
    not the day the session happened to be saved); the time of saving is used only when it is unknown."""
    now = when or datetime.datetime.now()
    pa  = wl_results["per_angle"]

    # Compute mean field sizes across all 4 angles
    mlc_vals, jaw_vals = [], []
    for ir in image_results.values():
        fe = ir.get("field_edges", {})
        m  = fe.get("mlc_total_mm")
        j  = fe.get("jaw_total_mm")
        if m is not None:
            mlc_vals.append(m)
        if j is not None:
            jaw_vals.append(j)
    avg_mlc = float(np.mean(mlc_vals)) if mlc_vals else None
    avg_jaw = float(np.mean(jaw_vals)) if jaw_vals else None

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM wl_records WHERE machine_name=? AND date=? AND time=?",           # saving the same session again replaces its record instead of adding a second trend point
                 (machine, now.strftime("%Y-%m-%d"), now.strftime("%H:%M")))
    conn.execute(
        """INSERT INTO wl_records (
            date, time, machine_name, physicist_name,
            walk_circle_r, pass_fail,
            baseline_dx, baseline_dy, baseline_dz,
            raw_dx_G0,  raw_dy_G0,
            raw_dx_G90, raw_dy_G90,
            raw_dx_G180,raw_dy_G180,
            raw_dx_G270,raw_dy_G270,
            avg_mlc_mm, avg_jaw_mm, logs_used
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M"),
            machine, physicist,
            wl_results["walk_circle_r"],
            1 if wl_results["pass_fail"] else 0,
            wl_results["baseline_dx"],
            wl_results["baseline_dy"],
            wl_results["baseline_dz"],
            pa[0]["raw_dx"],   pa[0]["raw_dy"],
            pa[90]["raw_dx"],  pa[90]["raw_dy"],
            pa[180]["raw_dx"], pa[180]["raw_dy"],
            pa[270]["raw_dx"], pa[270]["raw_dy"],
            avg_mlc, avg_jaw, wl_results.get("logs_used", 0),
        ),
    )
    if pf_results:
        try:
            _save_pf(conn, pf_results, machine, physicist, when)
        except Exception:
            pass                                                   # a picket-fence record problem must never lose the Winston-Lutz record
    conn.commit()
    try:                      # the four images are now a saved session: "Check EPID folder" no longer offers them as new
        import wl_sources
        wl_sources.init_registry(conn)
        wl_sources.mark(conn, [ir.get("sop_uid") for ir in image_results.values()], machine, "processed")
    except Exception:
        pass
    conn.close()


# ── PDF report ────────────────────────────────────────────────────────────────

_BLUE_DARK  = colors.HexColor("#0d47a1")
_BLUE_LIGHT = colors.HexColor("#e3f2fd")
_GREEN      = colors.HexColor("#1b5e20")
_GREEN_LITE = colors.HexColor("#c8e6c9")
_RED        = colors.HexColor("#b71c1c")
_RED_LITE   = colors.HexColor("#ffcdd2")
_STRIPE     = colors.HexColor("#f5f5f5")
_BORDER     = colors.HexColor("#bdbdbd")


def _val_color(v_mm: float):
    """ReportLab color for a displacement magnitude."""
    if v_mm < TOLERANCE_MM / 2:
        return colors.HexColor("#2e7d32")
    if v_mm < TOLERANCE_MM:
        return colors.HexColor("#e65100")
    return colors.HexColor("#b71c1c")


def generate_pdf_report(
    wl_results: dict,
    image_results: dict,
    output_path: str,
    diag_fig_path: str | None = None,
    walk_fig_path: str | None = None,
    machine_name: str = MACHINE_NAME,
    physicist_name: str = "",
    dicom_date: "datetime.date | None" = None,
    dicom_time: "datetime.time | None" = None,
    pf_results: dict | None = None,
    pf_fig_path: str | None = None,
):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.70 * inch,
        leftMargin=0.70 * inch,
        topMargin=0.70 * inch,
        bottomMargin=0.70 * inch,
    )

    ss = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "WLTitle",
        parent=ss["Title"],
        fontSize=19,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    sub_style = ParagraphStyle(
        "WLSub",
        parent=ss["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=10,
    )
    h2_style = ParagraphStyle(
        "WLH2",
        parent=ss["Heading2"],
        fontSize=11,
        spaceBefore=8,
        spaceAfter=3,
        textColor=_BLUE_DARK,
    )
    note_style = ParagraphStyle(
        "WLNote",
        parent=ss["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#333333"),
    )
    footer_style = ParagraphStyle(
        "WLFooter",
        parent=ss["Normal"],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#999999"),
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph(f"{APP_NAME} Report", title_style))
    story.append(Paragraph(f"{machine_name}  —  {PHANTOM_NAME}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=_BORDER, spaceAfter=8))

    # ── Metadata table ────────────────────────────────────────────────────────
    report_date = dicom_date    # always from DICOM / folder — never today's date
    report_time = dicom_time    # always from DICOM StudyTime
    today_str   = report_date.strftime("%B %d, %Y") if report_date else "Unknown Date"
    time_str    = report_time.strftime("%H:%M:%S") if report_time else "—"
    # Combined study datetime used everywhere (signature, footer, etc.)
    if report_date and report_time:
        study_dt = datetime.datetime.combine(report_date, report_time)
    elif report_date:
        study_dt = datetime.datetime.combine(report_date, datetime.time(0, 0))
    else:
        study_dt = datetime.datetime.now()
    max_walk  = wl_results["max_2d_walk_mm"]
    passed    = wl_results["pass_fail"]
    setup_lat = wl_results["setup_x"]    # patient lateral
    setup_si  = wl_results["setup_y"]    # patient SI
    setup_ap  = wl_results["setup_z"]    # patient AP
    setup_3d  = float(np.sqrt(setup_lat**2 + setup_si**2 + setup_ap**2))

    meta_rows = [
        ["Date:", today_str,           "Time:",        time_str],
        ["Machine:", machine_name,     "Phantom:",     PHANTOM_NAME],
        ["Physicist:", physicist_name, "Field Size:",  f"{FIELD_SIZE_MM:.0f}×{FIELD_SIZE_MM:.0f} mm"],
        ["Void Diameter:", f"{VOID_DIAMETER_MM:.1f} mm (air)",
         "Tolerance (max displacement):", f"≤ {TOLERANCE_MM:.1f} mm"],
        ["Machine log:", (f"applied to {wl_results['logs_used']} of 4 images (largest shift {wl_results['log_shift_max']:.2f} mm)"
                          if wl_results.get("logs_used") else "not used (no log for these images)"), "", ""],
        ["CBCT Setup Error (3D):",
         f"{setup_3d:.3f} mm  (Lat={setup_lat:+.3f},  SI={setup_si:+.3f},  AP={setup_ap:+.3f})",
         "", ""],
    ]
    meta_tbl = Table(meta_rows, colWidths=[1.55*inch, 1.85*inch, 1.2*inch, 2.5*inch])
    meta_tbl.setStyle(TableStyle([
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("FONTNAME",   (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",   (2, 0), (2, -1),  "Helvetica-Bold"),
        ("SPAN",       (1, 4), (3, 4)),   # machine-log line spans the last 3 cols
        ("SPAN",       (1, 5), (3, 5)),   # set-up error spans the last 3 cols
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, _STRIPE]),
        ("GRID",       (0, 0), (-1, -1), 0.3, _BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0.08 * inch))

    # ── Compute field-size status for banner (mirrors GUI logic) ─────────────
    _cfg_b   = _load_config()
    _frefs_b = _cfg_b.get("field_refs", {}).get(machine_name, {})
    _ref_mlc_b  = _frefs_b.get("mlc",  FIELD_SIZE_REF_MM)
    _ref_jaw_b  = _frefs_b.get("jaw",  FIELD_SIZE_REF_MM)
    _fs_calibrated = bool(_frefs_b)

    def _fs_lvl(dev, kind="mlc"):
        a = abs(dev)
        warn, fail = (FIELD_SIZE_JAW_WARN_MM, FIELD_SIZE_JAW_FAIL_MM) if kind == "jaw" \
            else (FIELD_SIZE_WARN_MM, FIELD_SIZE_FAIL_MM)
        if a <= warn: return 0
        if a <= fail: return 1
        return 2

    _mlc_devs_b, _jaw_devs_b = [], []
    for _ang in GANTRY_ANGLES:
        _fe = image_results[_ang].get("field_edges", {})
        _mv = _fe.get("mlc_total_mm")
        _jv = _fe.get("jaw_total_mm")
        if _mv is not None: _mlc_devs_b.append(_mv - _ref_mlc_b)
        if _jv is not None: _jaw_devs_b.append(_jv - _ref_jaw_b)

    _max_mlc_dev = max(_mlc_devs_b, key=abs) if _mlc_devs_b else None
    _max_jaw_dev = max(_jaw_devs_b, key=abs) if _jaw_devs_b else None
    _fs_worst = 0
    if _fs_calibrated:
        for _d in _mlc_devs_b:
            _fs_worst = max(_fs_worst, _fs_lvl(_d, "mlc"))
        for _d in _jaw_devs_b:
            _fs_worst = max(_fs_worst, _fs_lvl(_d, "jaw"))

    def _dsign(v): return f"{v:+.3f}" if v is not None else "—"

    if not _fs_calibrated:
        _fs_text  = "FIELD  —  Reference Not Set"
        _fs_bg    = colors.HexColor("#5a5a5a")
    elif _fs_worst == 0:
        _fs_text  = (f"FIELD  PASS    "
                     f"ΔMLC {_dsign(_max_mlc_dev)}  ΔJaw {_dsign(_max_jaw_dev)} mm")
        _fs_bg    = _GREEN
    elif _fs_worst == 1:
        _fs_text  = (f"FIELD  WARNING    "
                     f"ΔMLC {_dsign(_max_mlc_dev)}  ΔJaw {_dsign(_max_jaw_dev)} mm")
        _fs_bg    = colors.HexColor("#f57f17")
    else:
        _fs_text  = (f"FIELD  FAIL — Calibration Needed    "
                     f"ΔMLC {_dsign(_max_mlc_dev)}  ΔJaw {_dsign(_max_jaw_dev)} mm")
        _fs_bg    = _RED

    # ── WL PASS / FAIL banner (full width) ───────────────────────────────────
    status_text = "PASS" if passed else "FAIL"
    wl_bg       = _GREEN if passed else _RED
    wl_text     = f"WL  {status_text}    Isocenter Diameter = {max_walk:.3f} mm"

    banner_tbl = Table([[wl_text]], colWidths=[7.1 * inch])
    banner_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), wl_bg),
        ("TEXTCOLOR",     (0, 0), (-1, -1), colors.white),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 14),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 0.08 * inch))

    # ── Corrected displacements (left) + walk circle figure (right) ───────────
    corr_header = ["Gantry", "ΔLat/AP (mm)", "ΔSI (mm)", "|ΔR| (mm)"]
    corr_data   = [corr_header]
    last_row_bg = _GREEN_LITE if passed else _RED_LITE
    for angle in GANTRY_ANGLES:
        r     = wl_results["per_angle"][angle]
        dr    = np.sqrt(r["rel_dx"] ** 2 + r["rel_dy"] ** 2)
        x_dir = "AP" if angle in (90, 270) else "Lat"
        corr_data.append([
            f"G{angle:03d}° {x_dir}",
            f"{r['rel_dx']:+.3f}",
            f"{r['rel_dy']:+.3f}",
            f"{dr:.3f}",
        ])
    corr_data.append(["Max displacement", "", "", f"{max_walk:.3f}"])

    corr_tbl = Table(corr_data, colWidths=[1.20*inch, 1.00*inch, 0.95*inch, 0.95*inch])
    corr_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0),  (-1, 0),  _BLUE_DARK),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0),  (-1, -1), 9),
        ("ALIGN",         (1, 0),  (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -2), [colors.white, _BLUE_LIGHT]),
        ("BACKGROUND",    (0, -1), (-1, -1), last_row_bg),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID",          (0, 0),  (-1, -1), 0.3, _BORDER),
        ("TOPPADDING",    (0, 0),  (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0),  (-1, -1), 4),
        ("LEFTPADDING",   (0, 0),  (-1, -1), 5),
    ]))

    # Walk circle figure alongside the corrected table
    if walk_fig_path and os.path.exists(walk_fig_path):
        from PIL import Image as PILImage
        with PILImage.open(walk_fig_path) as _wfpil:
            _wf_pw, _wf_ph = _wfpil.size
        _wf_w = 2.95 * inch
        _wf_h = _wf_w * (_wf_ph / _wf_pw)
        _walk_cell = RLImage(walk_fig_path, width=_wf_w, height=_wf_h)
    else:
        _walk_cell = Paragraph("(figure unavailable)", note_style)

    _corr_label = Paragraph(
        "<b>Corrected Displacements — Isocenter Walk</b><br/>"
        f"<font size='8' color='#555555'>CBCT setup removed: "
        f"Lat {setup_lat:+.3f}  SI {setup_si:+.3f}  AP {setup_ap:+.3f} mm</font>",
        ParagraphStyle("CorrHdr", parent=ss["Normal"], fontSize=9,
                       spaceAfter=4, leading=13),
    )
    _no_pad = TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),
                          ("RIGHTPADDING",(0,0),(-1,-1),0),
                          ("TOPPADDING",(0,0),(-1,-1),0),
                          ("BOTTOMPADDING",(0,0),(-1,-1),0)])
    _corr_inner = Table([[_corr_label], [corr_tbl]], colWidths=[4.10 * inch])
    _corr_inner.setStyle(_no_pad)

    side_tbl = Table(
        [[_corr_inner, _walk_cell]],
        colWidths=[4.10 * inch, 3.00 * inch],
    )
    side_tbl.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ("ALIGN",        (1, 0), (1, 0),   "CENTER"),
    ]))
    story.append(side_tbl)
    story.append(Spacer(1, 0.07 * inch))

    # ── WL Methodology (iso page — after results) ─────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceAfter=5))
    story.append(Paragraph("Methodology Notes — Isocenter Walk", h2_style))
    story.append(Paragraph(
        "<b>Imaging &amp; Detection:</b>  "
        "Elekta iViewGT portal images (DICOM RTIMAGE, pydicom force=True) use an inverted "
        "pixel convention — LOW value = HIGH dose.  "
        "Pixel spacing is corrected to isocenter: spacing_iso = spacing_det × SAD/SID "
        "(Elekta Versa HD: 1000/1600 = 0.625).  "
        "Radiation field centre is the midpoint of 50%-penumbra crossings on column- and "
        "row-averaged profiles (~1024 px per profile, ≈32× noise reduction, sub-pixel via "
        "linear interpolation).  "
        f"The MIMI {VOID_DIAMETER_MM} mm air void (local pixel minimum within the field) is "
        "detected by Gaussian pre-filter (σ = void_r × 0.30), global minimum search within "
        f"{FIELD_SIZE_MM * VOID_SEARCH_HALF_FIELD_FRACTION:.0f} mm radius, then "
        "inverse-intensity-squared sub-pixel centroid.",
        note_style,
    ))
    story.append(Spacer(1, 0.04 * inch))
    story.append(Paragraph(
        "<b>3D Decomposition &amp; Walk Circle:</b>  "
        "The imager turns with the gantry, so a static phantom offset (X lateral, Y SI, Z AP) appears in the portal image as "
        "u = X cosG + Z sinG, v = Y (opposite signs at opposite gantry angles).  The offset that minimises the worst remaining "
        "residual over the four angles is removed; this is the centre of the smallest sphere touching the four beam axes.  The largest "
        "residual is the maximum displacement of any beam from the isocenter (TG-198 / MPPG)  "
        f"(PASS: ≤ {TOLERANCE_MM:.1f} mm, the SRS limit).",
        note_style,
    ))
    story.append(Spacer(1, 0.07 * inch))

    # ── Field size analysis  (page 2) ────────────────────────────────────────
    from reportlab.platypus import PageBreak as _PageBreak
    story.append(_PageBreak())
    story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceAfter=6))
    story.append(Paragraph(
        "Field Size Analysis  (Y-Jaw = MLC leaf banks,  X-Jaw = Physical collimator jaws)",
        h2_style,
    ))

    cfg   = _load_config()
    frefs = cfg.get("field_refs", {}).get(machine_name, {})
    ref_mlc  = frefs.get("mlc",  FIELD_SIZE_REF_MM)
    ref_jaw  = frefs.get("jaw",  FIELD_SIZE_REF_MM)
    ref_leaf = frefs.get("leaf", ref_mlc)

    # Field size PASS/FAIL banner
    fs_banner_tbl = Table([[_fs_text]], colWidths=[7.1 * inch])
    fs_banner_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _fs_bg),
        ("TEXTCOLOR",     (0, 0), (-1, -1), colors.white),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 13),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(fs_banner_tbl)
    story.append(Spacer(1, 0.06 * inch))

    _ref_note_style = ParagraphStyle("RefNote", parent=ss["Normal"], fontSize=8,
                                     textColor=colors.HexColor("#444444"), spaceAfter=4)
    story.append(Paragraph(
        f"Baselines (post Jun 19 2026):  MLC = {ref_mlc:.3f} mm  ·  "
        f"Jaw = {ref_jaw:.3f} mm  ·  Leaf span = {ref_leaf:.3f} mm.  "
        f"Action levels — MLC:  warn |Δ| > {FIELD_SIZE_WARN_MM:.1f} mm  ·  "
        f"call service |Δ| > {FIELD_SIZE_FAIL_MM:.1f} mm.  "
        f"Jaw:  warn |Δ| > {FIELD_SIZE_JAW_WARN_MM:.1f} mm  ·  "
        f"call service |Δ| > {FIELD_SIZE_JAW_FAIL_MM:.1f} mm.  "
        "Edges detected at 50%-penumbra crossing of column/row-averaged profiles (SAD/SID = 0.625).",
        _ref_note_style,
    ))

    _YEL2 = colors.HexColor("#f57f17")
    _RED2 = colors.HexColor("#b71c1c")
    _GRN2 = colors.HexColor("#2e7d32")

    def _fcc(v, ref, kind="mlc"):
        if v is None: return colors.HexColor("#888888")
        d = abs(v - ref)
        warn, fail = (FIELD_SIZE_JAW_WARN_MM, FIELD_SIZE_JAW_FAIL_MM) if kind == "jaw" \
            else (FIELD_SIZE_WARN_MM, FIELD_SIZE_FAIL_MM)
        return _GRN2 if d <= warn else (_YEL2 if d <= fail else _RED2)

    def _ffmt(v):       return f"{v:.3f}" if v is not None else "—"
    def _fdfmt(v, ref): return f"{v - ref:+.3f}" if v is not None else "—"

    # Per-angle table (narrowed to 4.1 in to sit beside leaf table)
    fs_hdr  = ["Gantry", "MLC (mm)", "Δ MLC", "Jaw (mm)", "Δ Jaw"]
    fs_data = [fs_hdr]
    mlc_acc, jaw_acc = [], []
    for angle in GANTRY_ANGLES:
        fe   = image_results[angle].get("field_edges", {})
        mlcv = fe.get("mlc_total_mm")
        jawv = fe.get("jaw_total_mm")
        dir_lbl = "AP" if angle in (90, 270) else "Lat"
        fs_data.append([f"G{angle:03d}° {dir_lbl}",
                        _ffmt(mlcv), _fdfmt(mlcv, ref_mlc),
                        _ffmt(jawv), _fdfmt(jawv, ref_jaw)])
        if mlcv is not None: mlc_acc.append(mlcv)
        if jawv is not None: jaw_acc.append(jawv)

    mean_mlc = float(np.mean(mlc_acc)) if mlc_acc else None
    mean_jaw = float(np.mean(jaw_acc)) if jaw_acc else None
    fs_data.append(["Mean", _ffmt(mean_mlc), _fdfmt(mean_mlc, ref_mlc),
                    _ffmt(mean_jaw), _fdfmt(mean_jaw, ref_jaw)])

    fs_col_w = [0.95*inch, 0.90*inch, 0.70*inch, 0.90*inch, 0.65*inch]   # 4.10 in total
    fs_tbl   = Table(fs_data, colWidths=fs_col_w)
    fs_cmds  = [
        ("BACKGROUND",    (0, 0),  (-1, 0),  _BLUE_DARK),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0),  (-1, -1), 9),
        ("ALIGN",         (1, 0),  (-1, -1), "CENTER"),
        ("BACKGROUND",    (0, -1), (-1, -1), _STRIPE),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID",          (0, 0),  (-1, -1), 0.3, _BORDER),
        ("TOPPADDING",    (0, 0),  (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0),  (-1, -1), 3),
        ("LEFTPADDING",   (0, 0),  (-1, -1), 4),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -2), [colors.white, _BLUE_LIGHT]),
    ]
    angle_rows = list(GANTRY_ANGLES) + [None]
    for ri, ang in enumerate(angle_rows):
        row_idx = ri + 1
        fe_row  = image_results.get(ang, {}).get("field_edges", {}) if ang is not None else {}
        mlcv    = fe_row.get("mlc_total_mm") if fe_row else mean_mlc
        jawv    = fe_row.get("jaw_total_mm") if fe_row else mean_jaw
        for ci, v, ref, kind in [(1, mlcv, ref_mlc, "mlc"), (2, mlcv, ref_mlc, "mlc"),
                                 (3, jawv, ref_jaw, "jaw"),  (4, jawv, ref_jaw, "jaw")]:
            fs_cmds.append(("TEXTCOLOR", (ci, row_idx), (ci, row_idx), _fcc(v, ref, kind)))
    fs_tbl.setStyle(TableStyle(fs_cmds))

    # Leaf span table (G000°), 3.0 in wide
    leaf_hdr_row = ["#", "SI (mm)", f"Span (mm)", f"Δ Span"]
    leaf_rows    = [leaf_hdr_row]
    leaves_g0    = image_results.get(0, {}).get("mlc_leaves", [])
    for leaf in leaves_g0:
        li   = leaf["leaf_idx"]
        si_c = (MLC_LEAVES_PER_BANK / 2 - li - 0.5) * MLC_LEAF_WIDTH_MM
        tot  = leaf.get("total_mm")
        leaf_rows.append([str(li + 1), f"{si_c:+.0f}", _ffmt(tot), _fdfmt(tot, ref_leaf)])
    leaf_col_w = [0.40*inch, 0.70*inch, 0.95*inch, 0.95*inch]   # 3.00 in total
    leaf_tbl   = Table(leaf_rows, colWidths=leaf_col_w)
    leaf_cmds  = [
        ("BACKGROUND",    (0, 0), (-1, 0),  _BLUE_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.3, _BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, _BLUE_LIGHT]),
    ]
    for li, leaf in enumerate(leaves_g0):
        col = _fcc(leaf.get("total_mm"), ref_leaf)
        for ci in (2, 3):
            leaf_cmds.append(("TEXTCOLOR", (ci, li + 1), (ci, li + 1), col))
    leaf_tbl.setStyle(TableStyle(leaf_cmds))

    _fs_angle_label = Paragraph(
        f"<b>Per-Gantry Field Size</b>  (Y-Jaw = MLC  ·  X-Jaw = SI jaws)",
        ParagraphStyle("FSAngHdr", parent=ss["Normal"], fontSize=9,
                       fontName="Helvetica-Bold", textColor=_BLUE_DARK, spaceAfter=3),
    )
    _fs_leaf_label  = Paragraph(
        f"<b>MLC Leaf Spans — G000°</b>  ({MLC_LEAVES_PER_BANK} leaves × {MLC_LEAF_WIDTH_MM:.0f} mm)",
        ParagraphStyle("FSLeafHdr", parent=ss["Normal"], fontSize=9,
                       fontName="Helvetica-Bold", textColor=_BLUE_DARK, spaceAfter=3),
    )

    _fs_inner_l = Table([[_fs_angle_label], [fs_tbl]], colWidths=[4.10 * inch])
    _fs_inner_l.setStyle(_no_pad)
    _fs_inner_r = Table([[_fs_leaf_label],  [leaf_tbl]], colWidths=[3.00 * inch])
    _fs_inner_r.setStyle(_no_pad)

    fs_side_tbl = Table(
        [[_fs_inner_l, _fs_inner_r]],
        colWidths=[4.10 * inch, 3.00 * inch],
    )
    fs_side_tbl.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    story.append(fs_side_tbl)
    story.append(Spacer(1, 0.07 * inch))

    # ── Field size methodology ────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceAfter=5))
    story.append(Paragraph("Methodology Notes — Field Size", h2_style))
    story.append(Paragraph(
        "Field edges are located at the 50%-penumbra crossing of column-averaged (MLC / Y-jaw) "
        "and row-averaged (X-jaw) pixel profiles.  "
        "Profile averaging across the full image width/height (~1024 px) suppresses MV portal "
        "noise by ≈32× and gives sub-pixel edge resolution via linear interpolation.  "
        "Total field size = separation between opposing edges at isocenter (SAD/SID = 0.625).  "
        f"Individual MLC leaf spans measured on G000° using {MLC_LEAVES_PER_BANK} leaf pairs "
        f"× {MLC_LEAF_WIDTH_MM:.0f} mm SI pitch.  "
        f"Action levels — MLC: warn |Δ| > {FIELD_SIZE_WARN_MM:.1f} mm, "
        f"call service |Δ| > {FIELD_SIZE_FAIL_MM:.1f} mm.  "
        f"Physical jaw: warn |Δ| > {FIELD_SIZE_JAW_WARN_MM:.1f} mm, "
        f"call service |Δ| > {FIELD_SIZE_JAW_FAIL_MM:.1f} mm (institutional protocol).  "
        "Results recorded to SQLite trend database.",
        note_style,
    ))

    # ── Page 3 (optional): Picket Fence ──────────────────────────────────────
    if pf_results is not None:
        story.append(_PageBreak())
        pf_ok   = pf_results["pass_fail"]
        pf_maxd = pf_results["max_deviation_mm"]
        pf_rms  = pf_results["rms_deviation_mm"]
        pf_n    = pf_results["n_leaves"]
        pf_avgw = pf_avg_width_central_mm(pf_results)
        pf_col  = "#1a3d1e" if pf_ok else "#3d1a1a"
        pf_tcol = colors.HexColor("#66bb6a" if pf_ok else "#ef5350")

        story.append(Paragraph("Picket Fence QA — MLC Leaf Position Accuracy", h2_style))
        story.append(Spacer(1, 0.06 * inch))

        # PF PASS/FAIL banner
        pf_banner_tbl = Table(
            [[Paragraph(
                f"{'PASS ✓' if pf_ok else 'FAIL ✗'}  —  "
                f"Max |Δ| = {pf_maxd:.2f} mm   "
                f"(Tolerance ≤ {PF_TOLERANCE_MM:.1f} mm)",
                ParagraphStyle("PFBanner", parent=ss["Normal"],
                               fontSize=17, fontName="Helvetica-Bold",
                               textColor=pf_tcol, alignment=TA_CENTER),
            )]],
            colWidths=[7.10 * inch],
        )
        pf_banner_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor(pf_col)),
            ("ROUNDEDCORNERS", [8]),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(pf_banner_tbl)
        story.append(Spacer(1, 0.10 * inch))

        # Summary stats table
        pf_stats_data = [
            ["Leaves analysed", "Max |Δ| (mm)", "RMS (mm)", "Avg Width (mm)"],
            [str(pf_n), f"{pf_maxd:.3f}", f"{pf_rms:.3f}", f"{pf_avgw:.2f}"],
        ]
        pf_stats_tbl = Table(pf_stats_data, colWidths=[1.77 * inch] * 4)
        pf_stats_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2e3a4a")),
            ("BACKGROUND",    (0, 1), (-1, 1),  colors.HexColor("#1e2530")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.HexColor("#aaaaaa")),
            ("TEXTCOLOR",     (0, 1), (-1, 1),  colors.HexColor("#dcdcdc")),
            ("TEXTCOLOR",     (1, 1), (1, 1),   pf_tcol),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  10),
            ("FONTNAME",      (0, 1), (-1, 1),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 1), (-1, 1),  13),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#3a3a3a")),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story.append(pf_stats_tbl)
        story.append(Spacer(1, 0.10 * inch))

        # PF figure
        if pf_fig_path and os.path.exists(pf_fig_path):
            fig_w, fig_h = 7.10 * inch, 4.80 * inch
            story.append(RLImage(pf_fig_path, width=fig_w, height=fig_h))
            story.append(Spacer(1, 0.06 * inch))

        # Methodology note
        story.append(Paragraph(
            f"<b>Picket Fence methodology:</b>  A single 1 cm MLC gap is delivered at G0° "
            f"and imaged on the EPID at SID = 1600 mm.  The central {PF_LEAVES_TOTAL} "
            f"Agility leaves ({PF_LEAVES_TOTAL // 2} per bank, "
            f"{PF_LEAF_WIDTH_MM:.0f} mm wide each at isocenter) contribute a "
            f"{PF_LEAVES_TOTAL // 2 * PF_LEAF_WIDTH_MM:.0f} mm long picket.  "
            f"For each leaf the 50% penumbra edge is located on both sides; the leaf "
            f"centre is their midpoint.  Deviation = leaf centre − mean of all leaf centres "
            f"(removes any residual phantom setup offset).  "
            f"<b>PASS criterion: all leaf deviations within "
            f"±{PF_TOLERANCE_MM:.1f} mm (warning ≥ ±{PF_WARN_MM:.1f} mm).</b>",
            note_style,
        ))

    # ── Page 3/4: Portal images + signature ──────────────────────────────────
    story.append(_PageBreak())
    if diag_fig_path and os.path.exists(diag_fig_path):
        story.append(Paragraph("Portal Images — Field & Void Centre Detection", h2_style))
        _cap_style = ParagraphStyle("DiagCaption", parent=ss["Normal"], fontSize=8,
                                    textColor=colors.HexColor("#555555"), spaceAfter=4,
                                    leading=11)
        story.append(Paragraph(
            "Panels 1–4 show each gantry angle.  "
            "Cyan dotted crosshair = 50%-penumbra field centre (primary WL reference).  "
            "Green lines = measured field edges (Y1/Y2 MLC, X1/X2 jaw).  "
            "Coloured circle &amp; marker = detected MIMI void centre "
            f"({VOID_DIAMETER_MM} mm diameter at isocenter).  "
            "Yellow arrow = raw displacement (field centre → void centre).  "
            "White bar = 5 mm scale at isocenter.  "
            "Panel 5 (displacement map): coloured dots = raw void positions per angle; "
            "red dashed ring = 1.0 mm limit; white circle = smallest circle holding "
            "the four residuals (its radius = maximum displacement); blue cross = fitted isocenter.",
            _cap_style,
        ))
        from PIL import Image as PILImage
        with PILImage.open(diag_fig_path) as _pil:
            _px_w, _px_h = _pil.size
        _usable_w = 7.1 * inch
        story.append(RLImage(diag_fig_path, width=_usable_w,
                             height=_usable_w * (_px_h / _px_w)))
        story.append(Spacer(1, 0.10 * inch))

    # ── Electronic Signature ──────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceAfter=6))
    story.append(Paragraph("Electronic Signature", h2_style))

    sig_rows = [
        ["Machine:", machine_name,    "Study Date:", today_str],
        ["Physicist:", physicist_name, "Study Time:", time_str],
        ["WL Result:", ("PASS" if passed else "FAIL") +
         f"   —   Max displacement = {max_walk:.3f} mm",
         "Phantom:", PHANTOM_NAME],
        ["Field Result:", _fs_text, "", ""],
    ]
    sig_bg    = _GREEN_LITE if passed else _RED_LITE
    _fs_sig_bg = (_GREEN_LITE if _fs_worst == 0 else
                  colors.HexColor("#fff8e1") if _fs_worst == 1 else _RED_LITE)
    _fs_sig_color = (_GREEN if _fs_worst == 0 else
                     colors.HexColor("#f57f17") if _fs_worst == 1 else _RED)

    # PF row (only when PF data was acquired)
    _pf_row_idx = None
    if pf_results is not None:
        _pf_ok      = pf_results.get("pass_fail", False)
        _pf_maxd    = pf_results.get("max_deviation_mm", 0.0)
        _pf_n       = pf_results.get("n_leaves", 0)
        _pf_text    = (f"PASS — Max |Δ| = {_pf_maxd:.2f} mm  "
                       f"({_pf_n} leaves,  Tol ≤ {PF_TOLERANCE_MM:.1f} mm  "
                       f"Warn ≥ {PF_WARN_MM:.1f} mm)"
                       if _pf_ok else
                       f"FAIL — Max |Δ| = {_pf_maxd:.2f} mm  "
                       f"({_pf_n} leaves,  Tol ≤ {PF_TOLERANCE_MM:.1f} mm)")
        _pf_warn    = (_pf_maxd >= PF_WARN_MM and _pf_maxd < PF_TOLERANCE_MM)
        _pf_sig_bg  = (_GREEN_LITE if _pf_ok and not _pf_warn else
                       colors.HexColor("#fff8e1") if _pf_warn else _RED_LITE)
        _pf_sig_col = (_GREEN if _pf_ok and not _pf_warn else
                       colors.HexColor("#f57f17") if _pf_warn else _RED)
        _pf_row_idx = len(sig_rows)
        sig_rows.append(["PF Result:", _pf_text, "", ""])

    sig_tbl = Table(sig_rows, colWidths=[1.0*inch, 2.7*inch, 1.1*inch, 2.3*inch])
    sig_style = [
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("FONTNAME",      (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -2),  "Helvetica-Bold"),
        ("FONTNAME",      (1, 2), (1, 2),   "Helvetica-Bold"),
        ("TEXTCOLOR",     (1, 2), (1, 2),   _GREEN if passed else _RED),
        ("BACKGROUND",    (0, 2), (-1, 2),  sig_bg),
        ("SPAN",          (1, 3), (3, 3)),
        ("FONTNAME",      (1, 3), (1, 3),   "Helvetica-Bold"),
        ("TEXTCOLOR",     (1, 3), (1, 3),   _fs_sig_color),
        ("BACKGROUND",    (0, 3), (-1, 3),  _fs_sig_bg),
        ("ROWBACKGROUNDS",(0, 0), (-1, 1),  [colors.white, _STRIPE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, _BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]
    if _pf_row_idx is not None:
        sig_style += [
            ("SPAN",       (1, _pf_row_idx), (3, _pf_row_idx)),
            ("FONTNAME",   (1, _pf_row_idx), (1, _pf_row_idx), "Helvetica-Bold"),
            ("TEXTCOLOR",  (1, _pf_row_idx), (1, _pf_row_idx), _pf_sig_col),
            ("BACKGROUND", (0, _pf_row_idx), (-1, _pf_row_idx), _pf_sig_bg),
        ]
    sig_tbl.setStyle(TableStyle(sig_style))
    story.append(sig_tbl)
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(
        f"This report was generated electronically by the {APP_NAME} Tool.  "
        f"The physicist named above ({physicist_name}) reviewed and approved the results "
        f"for the study performed on {study_dt.strftime('%B %d, %Y at %H:%M:%S')}.  "
        "This electronic signature is consistent with 21 CFR Part 11 intent for "
        "medical physics QA documentation.",
        ParagraphStyle("SigNote", parent=ss["Normal"], fontSize=8,
                       textColor=colors.HexColor("#555555"), leading=11),
    ))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.12 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceAfter=4))
    story.append(Paragraph(
        f"{APP_NAME} v1.0  —  "
        f"Study Date {study_dt.strftime('%Y-%m-%d %H:%M:%S')}  —  "
        f"{machine_name}  /  {PHANTOM_NAME}",
        footer_style,
    ))

    # Set PDF internal creation/modification dates to study date (not generation time).
    # In reportlab 4.x, PDFInfo.format() reads document._timeStamp; replace it with
    # a fake TimeStamp whose YMDhms/dhh/dmm match the DICOM study datetime.
    import time as _time
    _study_lt  = _time.localtime(study_dt.timestamp())
    _study_gmoff = getattr(_study_lt, "tm_gmtoff", 0)

    class _StudyTS:
        YMDhms = tuple(_study_lt)[:6]
        dhh    = int(_study_gmoff / 3600)
        dmm    = (_study_gmoff % 3600) // 60

    def _set_pdf_dates(canvas, _doc):
        canvas._doc._timeStamp = _StudyTS()

    doc.build(story, onFirstPage=_set_pdf_dates, onLaterPages=_set_pdf_dates)

    # Set filesystem mtime/atime to study date so file manager also shows study date
    _study_ts = study_dt.timestamp()
    os.utime(output_path, (_study_ts, _study_ts))

    print(f"PDF report written to: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _apply_dark_theme(app)
    window = WLApp()
    window.show()
    sys.exit(app.exec())
