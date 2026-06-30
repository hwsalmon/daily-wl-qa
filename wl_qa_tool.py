#!/usr/bin/env python3
"""
Daily Winston-Lutz QA Tool — Elekta Versa HD / Standard Imaging MIMI Phantom

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
import sqlite3
import datetime
import numpy as np
from pathlib import Path

# ── Dependency checks ─────────────────────────────────────────────────────────
_missing = []
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QComboBox, QTabWidget, QFrame, QDialog, QScrollArea,
        QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy,
        QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem,
        QHeaderView,
    )
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap, QIcon, QColor, QBrush
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
TOLERANCE_MM        = 1.0    # Maximum 2D isocenter walk (PASS/FAIL threshold)
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
FIELD_SIZE_WARN_MM  = 0.4   # |deviation| > this → orange (warning)
FIELD_SIZE_FAIL_MM  = 0.6   # |deviation| > this → red   (call service)

# ── Phantom ring constants ─────────────────────────────────────────────────────
# The MIMI phantom has an internal cylindrical feature whose shadow appears as a
# ring at ~22 mm radius in portal images. Ellipticity (horizontal vs vertical
# apparent radius) indicates phantom tilt: θ = arccos(r_short / r_long).
# ±45° sectors are used so that both cardinal and diagonal ring positions (where
# the ring is clearly inside the 40 mm square field) contribute to each axis.
RING_SEARCH_MIN_MM   = 10.0   # inner search boundary for phantom ring (mm)
RING_SEARCH_MAX_MM   = 24.0   # outer search boundary — below penumbra at ~25 mm
RING_GAUSSIAN_SIGMA  = 6.0    # smoothing σ (pixels) — suppresses MV portal noise
RING_RADIUS_NOMINAL_MM = 22.0 # expected ring radius (mm at iso)

MACHINE_NAME = "Elekta Versa HD"
PHANTOM_NAME = "Standard Imaging MIMI"

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

DB_PATH     = Path(__file__).parent / "wl_qa_history.db"
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

def load_dicom_images(directory: str) -> dict:
    """
    Scan directory (recursively) for DICOM files, map each to the nearest cardinal
    gantry angle.  Returns {0: ds, 90: ds, 180: ds, 270: ds}.
    Raises ValueError if any cardinal angle is missing.

    Uses force=True to handle Elekta RTIMAGE files that lack a standard DICOM
    File Meta Information header (common on older Elekta iViewGT exports).
    """
    root = Path(directory)
    # Collect every file in the tree — named *.dcm / *.DCM / or bare UID files
    dcm_files = (
        list(root.rglob("*.dcm"))
        + list(root.rglob("*.DCM"))
        + list(root.rglob("*.dicom"))
        + [f for f in root.rglob("*") if f.is_file()
           and f.suffix == "" and f not in
           (list(root.rglob("*.dcm")) + list(root.rglob("*.DCM")))]
    )
    # Deduplicate while preserving order
    seen = set()
    dcm_files = [f for f in dcm_files if not (f in seen or seen.add(f))]

    if not dcm_files:
        raise ValueError(f"No files found in:\n{directory}")

    images = {}
    skipped = []
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
            images[cardinal] = ds
        except Exception as e:
            skipped.append(f"{f.name} ({e})")

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


def find_void_center(
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


# ── Field size measurement ────────────────────────────────────────────────────

def _find_field_edge_pair(profile: np.ndarray) -> tuple:
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
    return left_edge, right_edge


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

    y_left_px, y_right_px = _find_field_edge_pair(h_profile)   # MLC
    x_top_px,  x_bot_px   = _find_field_edge_pair(v_profile)   # Jaw

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
        y_lp, y_rp = _find_field_edge_pair(profile)
        leaves.append({
            "leaf_idx":  i,
            "total_mm":  (y_rp - y_lp) * spacing_mm
                         if (y_lp is not None and y_rp is not None) else None,
        })
    return leaves


def measure_phantom_ring(
    arr: np.ndarray,
    spacing_mm: float,
    field_center_px: tuple,
) -> dict:
    """
    Measure the MIMI phantom internal ring radius at isocenter.

    The ring feature appears at ~22 mm radius in iViewGT portal images.
    Only the four diagonal sectors (45°/135°/225°/315° ± 15°) are used:
    at 45° the field corners are at ≥ 30 mm, well above the 24 mm search
    ceiling, so the field penumbra cannot contaminate the gradient peak.
    Cardinal and near-cardinal sectors are intentionally excluded because
    the MLC/Jaw penumbra at ~22–26 mm would overwhelm the ring gradient.

    Returns dict with r_mean_mm (ring radius in mm at iso).
    """
    fcx, fcy = field_center_px
    nr, nc = arr.shape

    r_max_px_crop = int(RING_SEARCH_MAX_MM / spacing_mm) + 5
    c0 = max(0, int(fcx) - r_max_px_crop)
    c1 = min(nc, int(fcx) + r_max_px_crop + 1)
    r0 = max(0, int(fcy) - r_max_px_crop)
    r1 = min(nr, int(fcy) + r_max_px_crop + 1)

    smooth  = gaussian_filter(arr[r0:r1, c0:c1].astype(np.float64),
                              sigma=RING_GAUSSIAN_SIGMA)
    fcx_loc = fcx - c0
    fcy_loc = fcy - r0

    C, Rm  = np.meshgrid(np.arange(smooth.shape[1], dtype=float),
                         np.arange(smooth.shape[0], dtype=float))
    dx_map = C - fcx_loc
    dy_map = Rm - fcy_loc
    dist   = np.sqrt(dx_map**2 + dy_map**2) + 1e-12

    r_min_px = int(RING_SEARCH_MIN_MM / spacing_mm)
    r_max_px = int(RING_SEARCH_MAX_MM / spacing_mm)
    r_range  = range(r_min_px, r_max_px + 1)

    def _ring_in_sector(dir_angle_deg: float, half_deg: float = 15.0) -> float | None:
        rad      = np.radians(dir_angle_deg)
        half_cos = np.cos(np.radians(half_deg))
        dot      = dx_map * np.cos(rad) + dy_map * np.sin(rad)
        sector   = (dot / dist) >= half_cos

        profile = np.full(len(r_range), np.nan)
        for i, r in enumerate(r_range):
            mask = (dist >= r - 0.5) & (dist < r + 0.5) & sector
            if mask.sum() >= 3:
                profile[i] = float(smooth[mask].mean())

        valid = ~np.isnan(profile)
        if valid.sum() < 5:
            return None
        x        = np.arange(len(profile))
        interp_p = np.interp(x, x[valid], profile[valid])
        grad     = np.abs(np.gradient(interp_p))
        peak     = int(np.argmax(grad))
        return (r_min_px + peak) * spacing_mm

    # Diagonal sectors only — furthest from field edges, free of penumbra contamination
    vals   = [v for a in [45, 135, 225, 315]
              if (v := _ring_in_sector(a)) is not None]
    r_mean = float(np.mean(vals)) if vals else None

    return {
        "r_mean_mm":      r_mean,
        "r_horiz_mm":     None,
        "r_vert_mm":      None,
        "ellipticity_mm": None,
        "tilt_deg":       None,
    }


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
    void_center_px = find_void_center(arr, spacing, search_fc)

    mlc_leaves = measure_mlc_leaves(arr, spacing)
    ring_info  = measure_phantom_ring(arr, spacing, search_fc)

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

    return {
        "gantry":               gantry,
        "spacing_mm":           spacing,
        "field_center_px":      primary_fc,            # edge midpoint (primary)
        "field_center_bbox_px": field_center_bbox_px,  # BBox (cross-check)
        "field_center_cent_px": field_center_cent_px,  # centroid (cross-check)
        "void_center_px":       void_center_px,
        "dx_mm":                dx_px_edge * spacing,  # PRIMARY
        "dy_mm":                dy_px_edge * spacing,
        "dx_mm_bbox":           dx_px_bbox * spacing,  # cross-check
        "dy_mm_bbox":           dy_px_bbox * spacing,
        "dx_mm_cent":           dx_px_cent * spacing,
        "dy_mm_cent":           dy_px_cent * spacing,
        "field_edges":          field_edges,
        "mlc_leaves":           mlc_leaves,
        "ring_info":            ring_info,
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


def compute_wl_results(
    image_results: dict,
    dx_key: str = "dx_mm",
    dy_key: str = "dy_mm",
) -> dict:
    """
    Compute 3D CBCT setup error, corrected walk residuals, and walk circle PASS/FAIL.

    Elekta iViewGT portal images are stored with a CONSISTENT patient coordinate
    orientation at all gantry angles (no coordinate flip at opposing angles).
    The portal-plane axes map to patient space as follows:
        G0°  / G180°  portal-X = patient lateral (L/R)
        G90° / G270°  portal-X = patient AP (A/P)
        All angles    portal-Y = patient SI (S/I)

    3D CBCT residual setup error is therefore:
        Lateral = mean(ΔX_G0, ΔX_G180)
        SI      = mean(ΔY_G0, ΔY_G90, ΔY_G180, ΔY_G270)
        AP      = mean(ΔX_G90, ΔX_G270)

    Subtracting the angle-appropriate component from each raw displacement yields
    pure mechanical walk residuals.  The Minimum Enclosing Circle of the four
    corrected walk vectors gives the walk circle metric.

    dx_key / dy_key select which displacement keys to read from each angle's dict,
    allowing comparison between field-center detection methods.
    """
    dx = {a: image_results[a][dx_key] for a in GANTRY_ANGLES}
    dy = {a: image_results[a][dy_key] for a in GANTRY_ANGLES}

    setup_x = (dx[0]  + dx[180]) / 2                               # patient lateral (mm)
    setup_y = (dy[0]  + dy[90] + dy[180] + dy[270]) / 4            # patient SI (mm)
    setup_z = (dx[90] + dx[270]) / 2                               # patient AP (mm)

    per_angle = {}
    for angle in GANTRY_ANGLES:
        x_corr = setup_z if angle in (90, 270) else setup_x        # lateral vs AP
        per_angle[angle] = {
            "raw_dx": dx[angle],
            "raw_dy": dy[angle],
            "rel_dx": dx[angle] - x_corr,
            "rel_dy": dy[angle] - setup_y,
        }

    corrected_pts = [(per_angle[a]["rel_dx"], per_angle[a]["rel_dy"]) for a in GANTRY_ANGLES]
    _, _, walk_r = _minimum_enclosing_circle(corrected_pts)

    return {
        "per_angle":      per_angle,
        "setup_x":        setup_x,    # patient lateral (mm)
        "setup_y":        setup_y,    # patient SI (mm)
        "setup_z":        setup_z,    # patient AP (mm)
        "baseline_dx":    setup_x,    # backward-compat alias
        "baseline_dy":    setup_y,
        "baseline_dz":    setup_z,
        "walk_circle_r":  walk_r,
        "max_2d_walk_mm": walk_r,
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
        f"Walk Circle Radius = {walk_r:.3f} mm   "
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

        # Window from void minimum to upper-penumbra level so both the air void
        # and the field penumbra / edge are clearly visible.
        # Elekta inverted: field = LOW value, background = HIGH value.
        # vmin anchors on the darkest pixels (void); vmax reaches well into the
        # background so the penumbra gradient appears as a visible ramp.
        crop_f = crop.flatten().astype(np.float64)
        vmin_c = float(np.percentile(crop_f, 0.5))
        vmax_c = float(np.percentile(crop_f, 82))

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

        # ── Phantom ring sanity circle (magenta) ─────────────────────────────
        ring  = r.get("ring_info", {})
        r_val = ring.get("r_mean_mm")
        if r_val is not None:
            ax.add_patch(mpatches.Circle(
                (fc[0], fc[1]), r_val / spc,
                edgecolor="magenta", facecolor="none",
                lw=1.2, ls="--", alpha=0.70, zorder=3,
            ))
            ax.text(
                0.98, 0.16,
                f"Ring  r={r_val:.1f} mm",
                transform=ax.transAxes,
                color="magenta", fontsize=6.0, va="bottom", ha="right",
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

    pts = [(per_angle[a]["rel_dx"], per_angle[a]["rel_dy"]) for a in GANTRY_ANGLES]
    mec_cx, mec_cy, _ = _minimum_enclosing_circle(pts)

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
        f"Tol {TOLERANCE_MM:.1f} mm", color="#ef5350", fontsize=7.5, zorder=3,
    )

    # MEC walk circle (white)
    ax.add_patch(mpatches.Circle(
        (mec_cx, mec_cy), walk_r, color="white", fill=False,
        lw=2.0, alpha=0.9, zorder=4,
    ))

    # MEC centre cross
    ax.plot(mec_cx, mec_cy, "+", color="#90caf9", ms=14, mew=2.5, zorder=6,
            label=f"MEC centre ({mec_cx:+.3f}, {mec_cy:+.3f}) mm")

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
        f"Isocenter Walk Circle   r = {walk_r:.3f} mm   "
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
        f"r = {walk_r:.3f} mm",
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
    n_leaves      = max(1, round(field_px_rows / leaf_px))

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


class WLApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Winston-Lutz Daily QA  —  Elekta Versa HD / MIMI Phantom")
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

        title_lbl = QLabel("Winston-Lutz Daily QA")
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

        self._pf_load_btn = QPushButton("Load Picket Fence")
        self._pf_load_btn.setMinimumWidth(170)
        self._pf_load_btn.clicked.connect(self._load_pf_directory)
        top_layout.addWidget(self._pf_load_btn)

        main_layout.addWidget(top_bar)

        # ── Selector bar ──────────────────────────────────────────────────────
        sel_bar = QWidget()
        sel_layout = QHBoxLayout(sel_bar)
        sel_layout.setContentsMargins(0, 0, 0, 0)

        machine_lbl = QLabel("Machine:")
        machine_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        sel_layout.addWidget(machine_lbl)

        self._machine_combo = QComboBox()
        self._machine_combo.addItems(MACHINES)
        self._machine_combo.setMinimumWidth(240)
        self._machine_combo.currentIndexChanged.connect(self._update_field_ref_labels)
        sel_layout.addWidget(self._machine_combo)
        sel_layout.addSpacing(24)

        physicist_lbl = QLabel("Physicist:")
        physicist_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        sel_layout.addWidget(physicist_lbl)

        self._physicist_combo = QComboBox()
        self._physicist_combo.addItems(PHYSICISTS)
        self._physicist_combo.setMinimumWidth(300)
        sel_layout.addWidget(self._physicist_combo)
        sel_layout.addStretch()

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
            f"Warn >{FIELD_SIZE_WARN_MM:.1f} mm     Fail >{FIELD_SIZE_FAIL_MM:.1f} mm"
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
            f"Walk Circle Radius: —   (Tolerance ≤ {TOLERANCE_MM:.1f} mm)"
        )
        self._walk_label.setStyleSheet("font-size: 15px;")
        left_vbox.addWidget(self._walk_label)

        self._dlg_status_label = QLabel("DLG Field Size:  —")
        self._dlg_status_label.setStyleSheet("font-size: 13px; color: #888888;")
        left_vbox.addWidget(self._dlg_status_label)

        self._ring_status_label = QLabel("Phantom Ring:  —")
        self._ring_status_label.setStyleSheet("font-size: 13px; color: #888888;")
        left_vbox.addWidget(self._ring_status_label)

        bottom_layout.addWidget(left_status)
        bottom_layout.addStretch()

        self._trends_btn = QPushButton("View Trends")
        self._trends_btn.setMinimumWidth(150)
        self._trends_btn.clicked.connect(self._show_trends)
        bottom_layout.addWidget(self._trends_btn)

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
            "— Jaw (always SI)  "
            f"— Warning >{FIELD_SIZE_WARN_MM:.1f} mm  "
            f"— Call Service >{FIELD_SIZE_FAIL_MM:.1f} mm"
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

        def _fs_level(dev: float) -> int:
            """0=pass, 1=warn, 2=fail"""
            a = abs(dev)
            if a <= FIELD_SIZE_WARN_MM:  return 0
            if a <= FIELD_SIZE_FAIL_MM:  return 1
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

        def _vcell_dev(v, ref):
            """Δ from reference cell — grey dash when reference not yet calibrated."""
            if not is_calibrated:
                return _cell("—", "#555555")
            dev   = (v - ref) if v is not None else None
            lvl   = _fs_level(dev) if dev is not None else -1
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
            self._fs_angle_table.setItem(ri, 2, _vcell_dev(jaw, ref_jaw))
            # Results tab: absolute + delta
            self._res_fs_angle_table.setItem(ri, 1, _cell_abs(mlc))
            self._res_fs_angle_table.setItem(ri, 2, _vcell_dev(mlc, ref_mlc))
            self._res_fs_angle_table.setItem(ri, 3, _cell_abs(jaw))
            self._res_fs_angle_table.setItem(ri, 4, _vcell_dev(jaw, ref_jaw))
            if is_calibrated:
                for v, ref in [(mlc, ref_mlc), (jaw, ref_jaw)]:
                    if v is not None:
                        worst_lvl = max(worst_lvl, _fs_level(v - ref))
            if mlc is not None:
                mlc_acc.append(mlc)
                mlc_devs.append(mlc - ref_mlc)
            if jaw is not None:
                jaw_acc.append(jaw)
                jaw_devs.append(jaw - ref_jaw)

        mean_mlc = float(np.mean(mlc_acc)) if mlc_acc else None
        mean_jaw = float(np.mean(jaw_acc)) if jaw_acc else None
        self._fs_angle_table.setItem(4, 1, _vcell_dev(mean_mlc, ref_mlc))
        self._fs_angle_table.setItem(4, 2, _vcell_dev(mean_jaw, ref_jaw))
        self._res_fs_angle_table.setItem(4, 1, _cell_abs(mean_mlc))
        self._res_fs_angle_table.setItem(4, 2, _vcell_dev(mean_mlc, ref_mlc))
        self._res_fs_angle_table.setItem(4, 3, _cell_abs(mean_jaw))
        self._res_fs_angle_table.setItem(4, 4, _vcell_dev(mean_jaw, ref_jaw))

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

    def _refresh_ring_display(self):
        """Update the phantom ring status label from the current image results."""
        if self._image_results is None:
            self._ring_status_label.setText("Phantom Ring:  —")
            self._ring_status_label.setStyleSheet("font-size: 13px; color: #888888;")
            return

        r_vals = []
        for ir in self._image_results.values():
            v = ir.get("ring_info", {}).get("r_mean_mm")
            if v is not None:
                r_vals.append(v)

        if not r_vals:
            self._ring_status_label.setText("Phantom Ring:  detection uncertain")
            self._ring_status_label.setStyleSheet("font-size: 13px; color: #888888;")
            return

        r_mean = float(np.mean(r_vals))
        in_range = 15.0 <= r_mean <= 30.0
        style = "font-size: 13px; color: #888888;" if in_range else \
                "font-size: 13px; color: #ffa726; font-weight: bold;"
        note  = "" if in_range else "  [unexpected — check phantom position]"
        self._ring_status_label.setText(
            f"Phantom Ring:  r = {r_mean:.1f} mm  (expected ≈{RING_RADIUS_NOMINAL_MM:.0f} mm){note}"
        )
        self._ring_status_label.setStyleSheet(style)

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

        self._dir_label.setText(Path(directory).name)
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
            image_results = {a: analyze_image(ds) for a, ds in dcm_images.items()}
            wl_results    = compute_wl_results(image_results)
            wl_cent = compute_wl_results(
                image_results, dx_key="dx_mm_cent", dy_key="dy_mm_cent"
            )
            wl_results["walk_circle_r_cent"] = wl_cent["walk_circle_r"]
            wl_results["pass_fail_cent"]     = wl_cent["pass_fail"]

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

            # Auto-detect picket fence image in the same directory
            self._pf_results  = None
            self._pf_fig_path = None
            for _pf_path in sorted(Path(directory).iterdir()):
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
            QMessageBox.critical(self, "Processing Error", str(exc))
            self._pf_label.setText("— ERROR —")
            self._pf_label.setStyleSheet(
                "font-size: 25px; font-weight: bold; color: #f44336; background: transparent;"
            )

    def _load_pf_directory(self):
        """Load a picket fence DICOM from a directory or single file."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Picket Fence DICOM Directory",
            self._config.get("last_dicom_dir", str(Path.home())),
        )
        if not path:
            return
        try:
            found = False
            for candidate in sorted(Path(path).iterdir()):
                if not candidate.is_file():
                    continue
                try:
                    ds_pf = pydicom.dcmread(str(candidate), force=True)
                    if identify_pf_dicom(ds_pf):
                        self._pf_results  = analyze_picket_fence(ds_pf)
                        self._pf_fig_path = generate_pf_figure(self._pf_results)
                        found = True
                        break
                except Exception:
                    pass
            if not found:
                # Try loading every file as a PF image regardless of label
                for candidate in sorted(Path(path).iterdir()):
                    if not candidate.is_file():
                        continue
                    try:
                        ds_pf = pydicom.dcmread(str(candidate), force=True)
                        if getattr(ds_pf, "Modality", "") == "RTIMAGE":
                            self._pf_results  = analyze_picket_fence(ds_pf)
                            self._pf_fig_path = generate_pf_figure(self._pf_results)
                            found = True
                            break
                    except Exception:
                        pass
            if not found:
                QMessageBox.warning(self, "Picket Fence",
                                    "No RTIMAGE DICOM found in that directory.")
                return
            self._refresh_pf_display()
        except Exception as exc:
            QMessageBox.critical(self, "Picket Fence Error", str(exc))

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
        avg_w = float(np.mean(pf["leaf_widths_mm"])) if len(pf["leaf_widths_mm"]) else 0.0

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
                f"Walk Circle:  Edge = {walk:.3f} mm  │  Centroid = {walk_cent:.3f} mm"
                f"   (Tolerance ≤ {TOLERANCE_MM:.1f} mm)"
            )
        else:
            self._walk_label.setText(
                f"Walk Circle Radius: {walk:.3f} mm   "
                f"(Tolerance ≤ {TOLERANCE_MM:.1f} mm)"
            )
        sx = wl["setup_x"]; sy = wl["setup_y"]; sz = wl["setup_z"]
        self._corr_sub.setText(
            f"CBCT setup: Lat={sx:+.3f} mm   SI={sy:+.3f} mm   AP={sz:+.3f} mm"
        )
        self._corr_sub.setStyleSheet("font-size: 12px; color: #90caf9; background: transparent;")

        def _color(v):
            if v < 0.5:
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
        self._refresh_ring_display()
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
        initial_path = str(
            Path(self._config.get("last_report_dir", str(Path.home()))) / default_name
        )
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Daily QA Report",
            initial_path,
            "PDF files (*.pdf)",
        )
        if not save_path:
            return
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
            _save_to_db(self._wl_results, self._image_results, machine, physicist)
            QMessageBox.information(self, "Report Saved", f"Report saved to:\n{save_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Report Error", str(exc))

    def _show_trends(self):
        """Open a dialog showing walk-circle-radius trend per machine."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Winston-Lutz Trend Analysis")
        dlg.resize(1000, 700)
        dlg_layout = QVBoxLayout(dlg)

        try:
            conn = sqlite3.connect(DB_PATH)
            cur  = conn.cursor()
            cur.execute(
                "SELECT date, machine_name, walk_circle_r, pass_fail "
                "FROM wl_records ORDER BY date ASC"
            )
            rows = cur.fetchall()
            conn.close()
        except Exception as exc:
            lbl = QLabel(f"Database error: {exc}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dlg_layout.addWidget(lbl)
            dlg.exec()
            return

        if not rows:
            lbl = QLabel(
                "No records in database yet.\n"
                "Generate a report to save the first record."
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 15px;")
            dlg_layout.addWidget(lbl)
            dlg.exec()
            return

        from collections import defaultdict
        machine_data: dict = defaultdict(list)
        for date_str, mname, walk_r, pf in rows:
            machine_data[mname].append((date_str, walk_r, bool(pf)))

        machine_colors = ["#64b5f6", "#81c784", "#ffb74d", "#f06292", "#ce93d8"]

        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#111111")
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        ax.set_xlabel("Date", color="white", fontsize=11)
        ax.set_ylabel("Walk Circle Radius (mm)", color="white", fontsize=11)
        ax.set_title(
            "Winston-Lutz Walk Circle Radius — Trend by Machine",
            color="white", fontsize=13, fontweight="bold",
        )

        ax.axhline(TOLERANCE_MM, color="#ef5350", linewidth=1.5, linestyle="--",
                   label=f"Tolerance ≤ {TOLERANCE_MM:.1f} mm")
        ax.axhline(0.5, color="#ffa726", linewidth=1.0, linestyle=":",
                   label="0.5 mm advisory")

        for idx, (mname, pts) in enumerate(sorted(machine_data.items())):
            col    = machine_colors[idx % len(machine_colors)]
            dates  = [p[0] for p in pts]
            vals   = [p[1] for p in pts]
            passes = [p[2] for p in pts]
            ax.plot(dates, vals, color=col, linewidth=1.5, marker="o",
                    markersize=6, label=mname)
            for x, y, ok in zip(dates, vals, passes):
                ax.scatter([x], [y], color="#66bb6a" if ok else "#ef5350",
                           s=55, zorder=5, edgecolors=col, linewidths=1)

        ax.legend(facecolor="#2b2b2b", edgecolor="#555555", labelcolor="white",
                  fontsize=9, loc="upper left")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8,
                 color="white")
        fig.tight_layout()

        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=120, bbox_inches="tight",
                    facecolor="#111111")
        plt.close(fig)
        buf.seek(0)

        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        chart_label = QLabel()
        chart_label.setPixmap(pixmap)
        chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dlg_layout.addWidget(chart_label, stretch=1)

        try:
            conn = sqlite3.connect(DB_PATH)
            cur  = conn.cursor()
            cur.execute(
                "SELECT date, machine_name, physicist_name, walk_circle_r, pass_fail "
                "FROM wl_records ORDER BY date DESC"
            )
            all_rows = cur.fetchall()
            conn.close()
        except Exception:
            all_rows = []

        hdrs = ["Date", "Machine", "Physicist", "Walk r (mm)", "Result"]
        table = QTableWidget(len(all_rows), len(hdrs))
        table.setHorizontalHeaderLabels(hdrs)
        table.setMaximumHeight(200)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)

        for ri, row in enumerate(all_rows):
            date_s, mname, phys, walk_r, pf = row
            vals_disp = [date_s, mname, phys, f"{walk_r:.3f}",
                         "PASS" if pf else "FAIL"]
            for c, val in enumerate(vals_disp):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == 4:
                    item.setForeground(
                        QBrush(QColor("#66bb6a" if pf else "#ef5350"))
                    )
                table.setItem(ri, c, item)

        dlg_layout.addWidget(table)
        dlg.exec()


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
    ]:
        try:
            conn.execute(col_def)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def _save_to_db(wl_results: dict, image_results: dict, machine: str, physicist: str):
    """Insert one QA session record into the database."""
    now = datetime.datetime.now()
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
    conn.execute(
        """INSERT INTO wl_records (
            date, time, machine_name, physicist_name,
            walk_circle_r, pass_fail,
            baseline_dx, baseline_dy, baseline_dz,
            raw_dx_G0,  raw_dy_G0,
            raw_dx_G90, raw_dy_G90,
            raw_dx_G180,raw_dy_G180,
            raw_dx_G270,raw_dy_G270,
            avg_mlc_mm, avg_jaw_mm
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
            avg_mlc, avg_jaw,
        ),
    )
    conn.commit()
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
    if v_mm < 0.5:
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
    story.append(Paragraph("Daily Winston-Lutz QA Report", title_style))
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
         "Tolerance:", f"≤ {TOLERANCE_MM:.1f} mm"],
        ["CBCT Setup Error (3D):",
         f"{setup_3d:.3f} mm  (Lat={setup_lat:+.3f},  SI={setup_si:+.3f},  AP={setup_ap:+.3f})",
         "", ""],
    ]
    meta_tbl = Table(meta_rows, colWidths=[1.55*inch, 1.85*inch, 1.2*inch, 2.5*inch])
    meta_tbl.setStyle(TableStyle([
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("FONTNAME",   (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",   (2, 0), (2, -1),  "Helvetica-Bold"),
        ("SPAN",       (1, 4), (3, 4)),   # MEC value spans last 3 cols
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

    def _fs_lvl(dev):
        a = abs(dev)
        if a <= FIELD_SIZE_WARN_MM: return 0
        if a <= FIELD_SIZE_FAIL_MM: return 1
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
        for _d in _mlc_devs_b + _jaw_devs_b:
            _fs_worst = max(_fs_worst, _fs_lvl(_d))

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
    wl_text     = f"WL  {status_text}    Walk Circle Radius = {max_walk:.3f} mm"

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
    corr_data.append(["Walk Circle r", "", "", f"{max_walk:.3f}"])

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
        "Portal axes: G0°/G180° X = Lateral; G90°/G270° X = AP; all Y = SI.  "
        "CBCT residual:  Lat = (ΔX_G0 + ΔX_G180)/2,  SI = mean(ΔY all angles),  "
        "AP = (ΔX_G90 + ΔX_G270)/2.  "
        "Subtracting the angle-appropriate component gives per-angle mechanical isocenter "
        "walk residuals.  The <b>Minimum Enclosing Circle (MEC)</b> radius of the four "
        f"residual vectors is the walk circle metric  "
        f"(PASS: ≤ {TOLERANCE_MM:.1f} mm, per AAPM TG-142 SRS/SBRT monthly tolerance).",
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
        f"Action levels:  warn |Δ| > {FIELD_SIZE_WARN_MM:.1f} mm  ·  "
        f"call service |Δ| > {FIELD_SIZE_FAIL_MM:.1f} mm.  "
        "Edges detected at 50%-penumbra crossing of column/row-averaged profiles (SAD/SID = 0.625).",
        _ref_note_style,
    ))

    _YEL2 = colors.HexColor("#f57f17")
    _RED2 = colors.HexColor("#b71c1c")
    _GRN2 = colors.HexColor("#2e7d32")

    def _fcc(v, ref):
        if v is None: return colors.HexColor("#888888")
        d = abs(v - ref)
        return _GRN2 if d <= FIELD_SIZE_WARN_MM else (_YEL2 if d <= FIELD_SIZE_FAIL_MM else _RED2)

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
        for ci, v, ref in [(1, mlcv, ref_mlc), (2, mlcv, ref_mlc),
                           (3, jawv, ref_jaw),  (4, jawv, ref_jaw)]:
            fs_cmds.append(("TEXTCOLOR", (ci, row_idx), (ci, row_idx), _fcc(v, ref)))
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
        f"Action levels: warn |Δ| > {FIELD_SIZE_WARN_MM:.1f} mm; "
        f"call service |Δ| > {FIELD_SIZE_FAIL_MM:.1f} mm (institutional protocol).  "
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
        pf_avgw = float(np.mean(pf_results["leaf_widths_mm"])) \
                  if len(pf_results["leaf_widths_mm"]) else 0.0
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
            "red dashed ring = 1.0 mm tolerance; white circle = MEC walk circle; "
            "blue cross = MEC centre.",
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
        ["Physicist:", physicist_name, "Study Time:",
         f"{time_str}"],
        ["WL Result:", ("PASS" if passed else "FAIL") +
         f"   —   Walk Circle = {max_walk:.3f} mm",
         "Phantom:", PHANTOM_NAME],
        ["Field Result:", _fs_text, "", ""],
    ]
    sig_tbl = Table(sig_rows, colWidths=[1.0*inch, 2.7*inch, 1.1*inch, 2.3*inch])
    sig_bg    = _GREEN_LITE if passed else _RED_LITE
    _fs_sig_bg = (_GREEN_LITE if _fs_worst == 0 else
                  colors.HexColor("#fff8e1") if _fs_worst == 1 else _RED_LITE)
    _fs_sig_color = (_GREEN if _fs_worst == 0 else
                     colors.HexColor("#f57f17") if _fs_worst == 1 else _RED)
    sig_tbl.setStyle(TableStyle([
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("FONTNAME",      (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -2),  "Helvetica-Bold"),
        ("FONTNAME",      (1, 2), (1, 2),   "Helvetica-Bold"),
        ("TEXTCOLOR",     (1, 2), (1, 2),   _GREEN if passed else _RED),
        ("BACKGROUND",    (0, 2), (-1, 2),  sig_bg),
        ("SPAN",          (1, 3), (3, 3)),                       # field result spans cols 1-3
        ("FONTNAME",      (1, 3), (1, 3),   "Helvetica-Bold"),
        ("TEXTCOLOR",     (1, 3), (1, 3),   _fs_sig_color),
        ("BACKGROUND",    (0, 3), (-1, 3),  _fs_sig_bg),
        ("ROWBACKGROUNDS",(0, 0), (-1, 1),  [colors.white, _STRIPE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, _BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(sig_tbl)
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(
        "This report was generated electronically by the Winston-Lutz QA Tool.  "
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
        f"Winston-Lutz QA Tool v1.0  —  "
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
