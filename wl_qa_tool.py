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
    from scipy.ndimage import gaussian_filter, label as scipy_label, sum as ndimage_sum
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

MACHINE_NAME = "Elekta Versa HD"
PHANTOM_NAME = "Standard Imaging MIMI"

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
    the normalised range, isolate the largest dark blob, and return its
    bounding-box geometric midpoint (more stable than centroid for flat-top fields).

    Returns (col_px, row_px) — i.e. (x, y) in image coordinates.
    """
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    # Field = DARK (low norm); background = BRIGHT (high norm)
    field_mask = (norm < 0.50).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, k)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN,  k)

    labeled, n = scipy_label(field_mask > 0)
    if n == 0:
        return arr.shape[1] / 2.0, arr.shape[0] / 2.0

    sizes = ndimage_sum(field_mask > 0, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    rows, cols = np.where(labeled == largest)

    col_center = (cols.min() + cols.max()) / 2.0
    row_center = (rows.min() + rows.max()) / 2.0
    return col_center, row_center


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


def analyze_image(ds) -> dict:
    """
    Full analysis of one EPID DICOM image.
    Returns field/void centres (pixels), pixel spacing, and raw displacement (mm).

    Sign convention:
      dx_mm > 0  →  void is to the RIGHT of field centre in the portal image
      dy_mm > 0  →  void is BELOW   field centre in the portal image
    """
    arr, spacing = get_pixel_array(ds)
    gantry = float(getattr(ds, "GantryAngle", 0.0))

    field_center_px = find_field_center(arr, spacing)
    void_center_px  = find_void_center(arr, spacing, field_center_px)

    dx_px = void_center_px[0] - field_center_px[0]
    dy_px = void_center_px[1] - field_center_px[1]

    return {
        "gantry":          gantry,
        "spacing_mm":      spacing,
        "field_center_px": field_center_px,
        "void_center_px":  void_center_px,
        "dx_mm":           dx_px * spacing,
        "dy_mm":           dy_px * spacing,
        "pixel_array":     arr,
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


def compute_wl_results(image_results: dict) -> dict:
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
    """
    dx = {a: image_results[a]["dx_mm"] for a in GANTRY_ANGLES}
    dy = {a: image_results[a]["dy_mm"] for a in GANTRY_ANGLES}

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

    # 5 panels: 4 portal images + 1 displacement / walk-circle map
    fig = plt.figure(figsize=(28, 6.4), facecolor="#111111", layout="constrained")
    gs  = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.1], wspace=0.07)
    portal_axes = [fig.add_subplot(gs[i]) for i in range(4)]
    disp_ax     = fig.add_subplot(gs[4])

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

        # Window to field-interior pixels only so the ~7% void contrast is visible.
        # Elekta inverted convention: field = LOW value, background = HIGH value.
        # A 0.15-normalised threshold separates field interior from penumbra/background.
        g_min = float(arr.min())
        g_max = float(arr.max())
        norm_crop = (crop.astype(np.float64) - g_min) / max(g_max - g_min, 1.0)
        field_mask = norm_crop < 0.15
        if field_mask.sum() > 200:
            fp     = crop[field_mask].astype(np.float64)
            vmin_c = float(np.percentile(fp, 2))
            vmax_c = float(np.percentile(fp, 90))
        else:
            vmin_c = float(np.percentile(crop, 0.5))
            vmax_c = float(np.percentile(crop, 60))

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

        # Field centre crosshairs
        ax.axhline(fc[1], color="cyan", lw=0.8, ls=":", alpha=0.7)
        ax.axvline(fc[0], color="cyan", lw=0.8, ls=":", alpha=0.7)

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

    # ── Walk-circle displacement map ──────────────────────────────────────────
    disp_ax.set_facecolor("#111111")
    for sp in disp_ax.spines.values():
        sp.set_color("#444444")
    disp_ax.tick_params(colors="#aaaaaa", labelsize=8)
    disp_ax.set_xlabel("ΔX (mm)  [Lat at G0°/G180°,  AP at G90°/G270°]",
                       color="#aaaaaa", fontsize=8)
    disp_ax.set_ylabel("ΔSI (mm)", color="#aaaaaa", fontsize=9)
    disp_ax.set_title(
        f"Void Displacement Map\n"
        f"Walk circle r = {walk_r:.3f} mm  "
        f"({'PASS' if passed else 'FAIL'})",
        color="#66bb6a" if passed else "#ef5350",
        fontsize=8.5, fontweight="bold",
    )

    # Grid
    disp_ax.axhline(0, color="#555555", lw=0.8)
    disp_ax.axvline(0, color="#555555", lw=0.8)
    disp_ax.grid(color="#333333", lw=0.5, ls="--")

    # Origin = ideal radiation isocenter
    disp_ax.plot(0, 0, "w+", ms=14, mew=2.0, zorder=6, label="Field ctr (ideal)")

    # Setup error (lateral shown on X axis; AP shown separately in title)
    ecx = wl_results["setup_x"]    # lateral
    ecy = wl_results["setup_y"]    # SI
    ecz = wl_results["setup_z"]    # AP
    disp_ax.plot(ecx, ecy, "w*", ms=10, zorder=6,
                 label=f"Setup Lat={ecx:+.2f} SI={ecy:+.2f} AP={ecz:+.2f}")

    # Walk circle
    theta = np.linspace(0, 2 * np.pi, 300)
    disp_ax.plot(
        ecx + walk_r * np.cos(theta),
        ecy + walk_r * np.sin(theta),
        color="white", lw=1.4, ls="-", alpha=0.6, label=f"Walk circle r={walk_r:.3f}mm",
    )

    # Tolerance circle centred on MEC centre
    disp_ax.plot(
        ecx + TOLERANCE_MM * np.cos(theta),
        ecy + TOLERANCE_MM * np.sin(theta),
        color="#888888", lw=1.0, ls=":", alpha=0.5, label=f"Tolerance {TOLERANCE_MM:.1f} mm",
    )

    # Raw displacement points per angle
    for angle in GANTRY_ANGLES:
        wla = wl_results["per_angle"][angle]
        col = angle_colors[angle]
        disp_ax.scatter(
            wla["raw_dx"], wla["raw_dy"],
            color=col, s=60, zorder=7,
        )
        disp_ax.annotate(
            f"G{angle}°",
            xy=(wla["raw_dx"], wla["raw_dy"]),
            xytext=(wla["raw_dx"] + 0.04, wla["raw_dy"] + 0.04),
            color=col, fontsize=7.5, fontweight="bold",
        )

    # Axis limits: at least ±(walk_r + 0.3) mm from MEC centre
    pad  = max(walk_r + 0.35, TOLERANCE_MM + 0.2)
    disp_ax.set_xlim(ecx - pad, ecx + pad)
    disp_ax.set_ylim(ecy - pad, ecy + pad)
    disp_ax.set_aspect("equal")
    disp_ax.legend(
        fontsize=6.5, loc="upper right",
        facecolor="#222222", edgecolor="#555555", labelcolor="white",
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return tmp.name


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
        self._dicom_date    = None
        self._table_labels: dict = {}

        self._config = _load_config()
        _init_db()
        self._build_ui()

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

        # ── PASS / FAIL banner ────────────────────────────────────────────────
        self._pf_frame = QFrame()
        self._pf_frame.setFixedHeight(85)
        self._pf_frame.setStyleSheet(
            "QFrame { background-color: #2b2b2b; border-radius: 10px; }"
        )
        pf_inner = QHBoxLayout(self._pf_frame)
        pf_inner.setContentsMargins(0, 0, 0, 0)

        self._pf_label = QLabel("—  AWAITING DATA  —")
        self._pf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pf_label.setStyleSheet(
            "font-size: 30px; font-weight: bold; color: gray; background: transparent;"
        )
        pf_inner.addWidget(self._pf_label)

        main_layout.addWidget(self._pf_frame)

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
        res_layout.setSpacing(16)
        self._tabview.addTab(tab_res, "Results")

        left_card = QFrame()
        left_card.setStyleSheet(
            "QFrame { background-color: #363636; border-radius: 10px; }"
        )
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(14, 14, 14, 12)

        raw_title = QLabel("Raw Displacements  (Field → Void)")
        raw_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        raw_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; background: transparent;"
        )
        left_layout.addWidget(raw_title)

        raw_sub = QLabel("Includes CBCT setup error")
        raw_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        raw_sub.setStyleSheet("font-size: 12px; color: gray; background: transparent;")
        left_layout.addWidget(raw_sub)

        raw_tbl_widget = QWidget()
        raw_tbl_widget.setStyleSheet("background: transparent;")
        self._build_result_table(raw_tbl_widget, prefix="raw")
        left_layout.addWidget(raw_tbl_widget, stretch=1)
        res_layout.addWidget(left_card)

        right_card = QFrame()
        right_card.setStyleSheet(
            "QFrame { background-color: #363636; border-radius: 10px; }"
        )
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(14, 14, 14, 12)

        corr_title = QLabel("Corrected Displacements  (Isocenter Walk)")
        corr_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        corr_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; background: transparent;"
        )
        right_layout.addWidget(corr_title)

        self._corr_sub = QLabel("Isocenter walk  (3D setup error removed)")
        self._corr_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._corr_sub.setStyleSheet("font-size: 12px; color: gray; background: transparent;")
        right_layout.addWidget(self._corr_sub)

        corr_tbl_widget = QWidget()
        corr_tbl_widget.setStyleSheet("background: transparent;")
        self._build_result_table(corr_tbl_widget, prefix="corr")
        right_layout.addWidget(corr_tbl_widget, stretch=1)
        res_layout.addWidget(right_card)

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

        self._walk_label = QLabel(
            f"Walk Circle Radius: —   (Tolerance ≤ {TOLERANCE_MM:.1f} mm)"
        )
        self._walk_label.setStyleSheet("font-size: 15px;")
        bottom_layout.addWidget(self._walk_label)
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
            "font-size: 30px; font-weight: bold; color: gray; background: transparent;"
        )
        self._pf_frame.setStyleSheet(
            "QFrame { background-color: #2b2b2b; border-radius: 10px; }"
        )
        QApplication.processEvents()

        try:
            dcm_images    = load_dicom_images(directory)
            image_results = {a: analyze_image(ds) for a, ds in dcm_images.items()}
            wl_results    = compute_wl_results(image_results)

            self._dicom_date = None
            for ds in dcm_images.values():
                for tag in ("AcquisitionDate", "StudyDate", "ContentDate"):
                    raw = getattr(ds, tag, None)
                    if raw and len(str(raw)) == 8:
                        try:
                            self._dicom_date = datetime.datetime.strptime(
                                str(raw), "%Y%m%d"
                            ).date()
                        except ValueError:
                            pass
                        break
                if self._dicom_date:
                    break

            self._image_results = image_results
            self._wl_results    = wl_results
            self._loaded_dir    = directory
            self._diag_fig_path = generate_diagnostic_figure(image_results, wl_results)

            self._refresh_display()
            self._report_btn.setEnabled(True)

        except Exception as exc:
            QMessageBox.critical(self, "Processing Error", str(exc))
            self._pf_label.setText("— ERROR —")
            self._pf_label.setStyleSheet(
                "font-size: 30px; font-weight: bold; color: #f44336; background: transparent;"
            )

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
            self._pf_label.setText(f"PASS    {walk:.2f} mm")
            self._pf_label.setStyleSheet(
                "font-size: 30px; font-weight: bold; color: #66bb6a; background: transparent;"
            )
        else:
            self._pf_frame.setStyleSheet(
                "QFrame { background-color: #3d1a1a; border-radius: 10px; }"
            )
            self._pf_label.setText(f"FAIL    {walk:.2f} mm")
            self._pf_label.setStyleSheet(
                "font-size: 30px; font-weight: bold; color: #ef5350; background: transparent;"
            )

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
            raw_dr  = np.sqrt(r["raw_dx"] ** 2 + r["raw_dy"] ** 2)
            corr_dr = np.sqrt(r["rel_dx"] ** 2 + r["rel_dy"] ** 2)

            _set(f"raw_{angle}_dx",  f"{r['raw_dx']:+.3f}", _color(abs(r["raw_dx"])))
            _set(f"raw_{angle}_dy",  f"{r['raw_dy']:+.3f}", _color(abs(r["raw_dy"])))
            _set(f"raw_{angle}_dr",  f"{raw_dr:.3f}",       _color(raw_dr))
            _set(f"corr_{angle}_dx", f"{r['rel_dx']:+.3f}", _color(abs(r["rel_dx"])))
            _set(f"corr_{angle}_dy", f"{r['rel_dy']:+.3f}", _color(abs(r["rel_dy"])))
            _set(f"corr_{angle}_dr", f"{corr_dr:.3f}",      _color(corr_dr))

        self._update_diag_image()

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

        default_name = (
            f"WL_QA_{machine.replace(' ', '_')}_"
            f"{datetime.date.today().strftime('%Y%m%d')}.pdf"
        )
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
                machine_name=machine,
                physicist_name=physicist,
                dicom_date=self._dicom_date,
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
    # Migrate older databases that lack baseline_dz
    try:
        conn.execute("ALTER TABLE wl_records ADD COLUMN baseline_dz REAL")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def _save_to_db(wl_results: dict, image_results: dict, machine: str, physicist: str):
    """Insert one QA session record into the database."""
    now   = datetime.datetime.now()
    pa    = wl_results["per_angle"]
    conn  = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO wl_records (
            date, time, machine_name, physicist_name,
            walk_circle_r, pass_fail,
            baseline_dx, baseline_dy, baseline_dz,
            raw_dx_G0,  raw_dy_G0,
            raw_dx_G90, raw_dy_G90,
            raw_dx_G180,raw_dy_G180,
            raw_dx_G270,raw_dy_G270
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
    machine_name: str = MACHINE_NAME,
    physicist_name: str = "",
    dicom_date: "datetime.date | None" = None,
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
        fontSize=12,
        spaceBefore=14,
        spaceAfter=4,
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
    report_date = dicom_date if dicom_date else datetime.date.today()
    today_str   = report_date.strftime("%B %d, %Y")
    time_str    = datetime.datetime.now().strftime("%H:%M")
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
    story.append(Spacer(1, 0.12 * inch))

    # ── PASS / FAIL banner ────────────────────────────────────────────────────
    status_text = "PASS" if passed else "FAIL"
    banner_bg   = _GREEN if passed else _RED
    pf_tbl = Table(
        [[f"OVERALL RESULT:  {status_text}       Walk Circle Radius = {max_walk:.3f} mm"]],
        colWidths=[7.1 * inch],
    )
    pf_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), banner_bg),
        ("TEXTCOLOR",     (0, 0), (-1, -1), colors.white),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 15),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(pf_tbl)
    story.append(Spacer(1, 0.14 * inch))

    # ── Raw displacements ─────────────────────────────────────────────────────
    story.append(Paragraph("Raw Displacements  (Field Centre → Void Centre)", h2_style))
    story.append(Paragraph(
        f"3D CBCT setup error: Lateral={setup_lat:+.3f} mm,  SI={setup_si:+.3f} mm,  AP={setup_ap:+.3f} mm  |  "
        f"Walk circle radius: {max_walk:.3f} mm",
        ParagraphStyle("mecnote", parent=ss["Normal"], fontSize=9,
                       textColor=colors.HexColor("#444444"), spaceAfter=4),
    ))

    raw_header = [
        "Gantry", "ΔX raw (mm)", "ΔSI raw (mm)", "|ΔR| raw (mm)", "ΔX represents"
    ]
    raw_data = [raw_header]
    for angle in GANTRY_ANGLES:
        r  = wl_results["per_angle"][angle]
        dr = np.sqrt(r["raw_dx"] ** 2 + r["raw_dy"] ** 2)
        x_dir = "AP (patient A/P)" if angle in (90, 270) else "Lateral (patient L/R)"
        raw_data.append([
            f"G{angle:03d}°",
            f"{r['raw_dx']:+.3f}",
            f"{r['raw_dy']:+.3f}",
            f"{dr:.3f}",
            x_dir,
        ])

    raw_tbl = Table(raw_data, colWidths=[0.9*inch, 1.2*inch, 1.2*inch, 1.3*inch, 2.5*inch])
    raw_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  _BLUE_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (1, 0), (3, -1),  "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, _BLUE_LIGHT]),
        ("FONTNAME",      (0, 2), (0, 2),   "Helvetica-Bold"),  # G0 row
        ("GRID",          (0, 0), (-1, -1), 0.3, _BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(raw_tbl)
    story.append(Spacer(1, 0.12 * inch))

    # ── Corrected displacements ───────────────────────────────────────────────
    story.append(Paragraph(
        "Corrected Displacements  (3D Setup Error Removed — Isocenter Walk)",
        h2_style,
    ))

    corr_header = [
        "Gantry", "ΔLat/AP corr (mm)", "ΔSI corr (mm)", "|ΔR| corr (mm)", "Status"
    ]
    corr_data = [corr_header]
    for angle in GANTRY_ANGLES:
        r  = wl_results["per_angle"][angle]
        dr = np.sqrt(r["rel_dx"] ** 2 + r["rel_dy"] ** 2)
        x_dir = "AP walk" if angle in (90, 270) else "Lat walk"
        status_cell = "✓  PASS" if dr <= TOLERANCE_MM else "✗  FAIL"
        corr_data.append([
            f"G{angle:03d}°  ({x_dir})",
            f"{r['rel_dx']:+.3f}",
            f"{r['rel_dy']:+.3f}",
            f"{dr:.3f}",
            status_cell,
        ])

    # Summary row
    corr_data.append([
        "WALK CIRCLE r", "", "", f"{max_walk:.3f}",
        "PASS" if passed else "FAIL",
    ])

    last_row_bg = _GREEN_LITE if passed else _RED_LITE
    corr_tbl = Table(corr_data, colWidths=[1.3*inch, 1.2*inch, 1.2*inch, 1.2*inch, 2.2*inch])
    corr_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0),  (-1, 0),  _BLUE_DARK),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0),  (-1, -1), 9),
        ("ALIGN",         (1, 0),  (3, -1),  "CENTER"),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -2), [colors.white, _BLUE_LIGHT]),
        ("BACKGROUND",    (0, -1), (-1, -1), last_row_bg),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID",          (0, 0),  (-1, -1), 0.3, _BORDER),
        ("TOPPADDING",    (0, 0),  (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0),  (-1, -1), 4),
        ("LEFTPADDING",   (0, 0),  (-1, -1), 6),
    ]))
    story.append(corr_tbl)
    story.append(Spacer(1, 0.14 * inch))

    # ── Diagnostic portal images ───────────────────────────────────────────────
    if diag_fig_path and os.path.exists(diag_fig_path):
        story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceAfter=6))
        story.append(Paragraph("Portal Images — Field & Void Centre Detection", h2_style))
        caption_style = ParagraphStyle(
            "DiagCaption",
            parent=ss["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#555555"),
            spaceAfter=4,
        )
        story.append(Paragraph(
            "Cyan dashed box = 40×40 mm radiation field boundary (field of view).  "
            "Cyan dotted crosshair = field centre (radiation isocenter).  "
            "Coloured + and circle = detected void centre; circle = 6.4 mm air void diameter at isocenter.  "
            "Yellow arrow = raw displacement vector (field centre → void centre).  "
            "White bar = 5 mm scale at isocenter.  "
            "5th panel: displacement map — coloured dots = raw void positions, white circle = MEC walk circle.",
            caption_style,
        ))
        # Scale image to full usable page width, preserving aspect ratio
        from PIL import Image as PILImage
        with PILImage.open(diag_fig_path) as pil_img:
            px_w, px_h = pil_img.size
        usable_w = 7.1 * inch
        aspect   = px_h / px_w
        story.append(RLImage(diag_fig_path, width=usable_w, height=usable_w * aspect))
        story.append(Spacer(1, 0.10 * inch))

    # ── Methodology notes ─────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceAfter=6))
    story.append(Paragraph("Methodology Notes", h2_style))
    story.append(Paragraph(
        "<b>Void Detection (Inverted Logic):</b>  The MIMI phantom isocenter target is a "
        f"{VOID_DIAMETER_MM} mm diameter air void embedded in acetal copolymer (ρ ≈ 1.41 g/cm³). "
        "At MV energies air attenuates less than acetal, so the void receives more dose and appears "
        "as the LOCAL MINIMUM pixel value in the field on Elekta iViewGT RTIMAGEs "
        "(inverted storage convention: LOW value = HIGH dose). "
        "Detection: Gaussian pre-filter (σ ≈ void_radius × 0.30 px); global minimum in the "
        f"{FIELD_SIZE_MM * VOID_SEARCH_HALF_FIELD_FRACTION:.0f} mm search window; "
        "inverse-intensity-squared centroid for sub-pixel localisation. "
        "Portal images are windowed to field-interior pixels only to make the ~7% void contrast visible.",
        note_style,
    ))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(
        "<b>Walk Circle (Minimum Enclosing Circle):</b>  "
        "Elekta iViewGT portal images are stored with a consistent patient coordinate orientation "
        "at all gantry angles (no image flip at opposing angles). "
        "The portal-plane axes map to patient space as: "
        "G0°/G180° portal-X = patient Lateral (L/R);  "
        "G90°/G270° portal-X = patient AP (A/P);  "
        "all angles portal-Y = patient SI (S/I).  "
        "The residual CBCT setup error in 3D patient coordinates is therefore: "
        "Lateral = (ΔX_G0 + ΔX_G180)/2,  "
        "SI = mean(ΔY_G0, ΔY_G90, ΔY_G180, ΔY_G270),  "
        "AP = (ΔX_G90 + ΔX_G270)/2.  "
        "Subtracting the angle-appropriate component from each raw displacement yields "
        "the per-angle isocenter walk residuals.  "
        "The Minimum Enclosing Circle radius of the four corrected walk vectors is the walk circle metric.",
        note_style,
    ))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(
        f"<b>Tolerance:</b>  Walk circle radius ≤ {TOLERANCE_MM:.1f} mm = PASS.  "
        "Pixel spacing corrected from detector plane to isocenter using "
        "SAD/SID magnification (spacing_iso = spacing_det × SAD/SID).",
        note_style,
    ))

    # ── Electronic Signature ──────────────────────────────────────────────────
    story.append(Spacer(1, 0.18 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceAfter=8))
    story.append(Paragraph("Electronic Signature", h2_style))

    sig_time = datetime.datetime.now()
    sig_rows = [
        ["Machine:", machine_name,    "Report Date:", today_str],
        ["Physicist:", physicist_name, "Signed:",
         f"{sig_time.strftime('%Y-%m-%d  %H:%M')}"],
        ["Result:", ("PASS" if passed else "FAIL") +
         f"   —   Walk Circle Radius = {max_walk:.3f} mm",
         "Phantom:", PHANTOM_NAME],
    ]
    sig_tbl = Table(sig_rows, colWidths=[1.0*inch, 2.7*inch, 1.1*inch, 2.3*inch])
    sig_bg   = _GREEN_LITE if passed else _RED_LITE
    sig_tbl.setStyle(TableStyle([
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("FONTNAME",      (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -1),  "Helvetica-Bold"),
        ("FONTNAME",      (1, 2), (1, 2),   "Helvetica-Bold"),
        ("TEXTCOLOR",     (1, 2), (1, 2),   _GREEN if passed else _RED),
        ("BACKGROUND",    (0, 2), (-1, 2),  sig_bg),
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
        f"by initiating report generation on {sig_time.strftime('%B %d, %Y at %H:%M')}.  "
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
        f"Generated {sig_time.strftime('%Y-%m-%d %H:%M')}  —  "
        f"{machine_name}  /  {PHANTOM_NAME}",
        footer_style,
    ))

    doc.build(story)
    print(f"PDF report written to: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _apply_dark_theme(app)
    window = WLApp()
    window.show()
    sys.exit(app.exec())
