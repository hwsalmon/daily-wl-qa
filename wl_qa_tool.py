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
subtracted from all cardinal-angle displacements to isolate true mechanical walk.
"""

import os
import sys
import datetime
import numpy as np
from pathlib import Path

# ── Dependency checks ─────────────────────────────────────────────────────────
_missing = []
try:
    import customtkinter as ctk
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError:
    _missing.append("customtkinter")

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
TOLERANCE_MM        = 1.0    # Maximum 2D mechanical walk (PASS/FAIL threshold)
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
    Compute setup-error-corrected displacements and walk circle PASS/FAIL.

    The residual CBCT setup error is estimated from ALL four images as the centre
    of the Minimum Enclosing Circle (MEC) of the four raw void-to-field-centre
    displacement vectors.  Subtracting the MEC centre from each raw vector gives
    the per-angle mechanical walk.  The MEC radius is the walk circle metric
    (smallest circle containing all four walk vectors) used for PASS/FAIL.

    Using all four angles to estimate setup error is more robust than the single
    G0° reference, and avoids the coordinate-flip ambiguity that affects the
    naive G0 subtraction at G180°.
    """
    raw = [(image_results[a]["dx_mm"], image_results[a]["dy_mm"]) for a in GANTRY_ANGLES]
    cx, cy, walk_r = _minimum_enclosing_circle(raw)

    per_angle = {}
    for angle in GANTRY_ANGLES:
        r = image_results[angle]
        per_angle[angle] = {
            "raw_dx": r["dx_mm"],
            "raw_dy": r["dy_mm"],
            "rel_dx": r["dx_mm"] - cx,
            "rel_dy": r["dy_mm"] - cy,
        }

    return {
        "per_angle":      per_angle,
        "baseline_dx":    cx,       # MEC centre X (setup error estimate)
        "baseline_dy":    cy,       # MEC centre Y
        "walk_circle_r":  walk_r,
        "max_2d_walk_mm": walk_r,   # used by GUI / PDF for PASS/FAIL banner
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
        ax.set_title(
            f"G{angle:03d}°  |  Raw |ΔR|={np.hypot(wla['raw_dx'],wla['raw_dy']):.3f} mm\n"
            f"Raw  ΔX={wla['raw_dx']:+.3f}  ΔY={wla['raw_dy']:+.3f} mm\n"
            f"Corr ΔX={wla['rel_dx']:+.3f}  ΔY={wla['rel_dy']:+.3f}  [{corr_label}]",
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
    disp_ax.set_xlabel("ΔX (mm)", color="#aaaaaa", fontsize=9)
    disp_ax.set_ylabel("ΔY (mm)", color="#aaaaaa", fontsize=9)
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

    # MEC centre = setup error estimate
    ecx = wl_results["baseline_dx"]
    ecy = wl_results["baseline_dy"]
    disp_ax.plot(ecx, ecy, "w*", ms=10, zorder=6, label=f"MEC ctr ({ecx:+.2f},{ecy:+.2f})")

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

class WLApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Winston-Lutz Daily QA  —  Elekta Versa HD / MIMI Phantom")
        self.geometry("960x680")
        self.minsize(820, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._wl_results    = None
        self._image_results = None
        self._loaded_dir    = None
        self._diag_fig_path = None
        self._table_labels: dict = {}

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(18, 8))

        ctk.CTkLabel(
            top,
            text="Winston-Lutz Daily QA",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(side="left")

        self._load_btn = ctk.CTkButton(
            top,
            text="Load DICOM Directory",
            command=self._load_directory,
            width=190,
            font=ctk.CTkFont(size=13),
        )
        self._load_btn.pack(side="right")

        self._dir_label = ctk.CTkLabel(
            top,
            text="No directory loaded",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self._dir_label.pack(side="right", padx=12)

        # ── PASS / FAIL banner ────────────────────────────────────────────────
        self._pf_frame = ctk.CTkFrame(self, corner_radius=10, height=72)
        self._pf_frame.pack(fill="x", padx=20, pady=4)
        self._pf_frame.pack_propagate(False)

        self._pf_label = ctk.CTkLabel(
            self._pf_frame,
            text="—  AWAITING DATA  —",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color="gray",
        )
        self._pf_label.pack(expand=True)

        # ── Results tables ────────────────────────────────────────────────────
        tables_row = ctk.CTkFrame(self, fg_color="transparent")
        tables_row.pack(fill="both", expand=True, padx=20, pady=6)

        left_card = ctk.CTkFrame(tables_row, corner_radius=10)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 8))

        ctk.CTkLabel(
            left_card,
            text="Raw Displacements  (Field → Void)",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(pady=(12, 2))
        ctk.CTkLabel(
            left_card,
            text="Includes CBCT setup error",
            font=ctk.CTkFont(size=10),
            text_color="gray",
        ).pack(pady=(0, 6))

        raw_tbl = ctk.CTkFrame(left_card, fg_color="transparent")
        raw_tbl.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self._build_result_table(raw_tbl, prefix="raw")

        right_card = ctk.CTkFrame(tables_row, corner_radius=10)
        right_card.pack(side="right", fill="both", expand=True, padx=(8, 0))

        ctk.CTkLabel(
            right_card,
            text="Corrected Displacements  (MEC setup error removed)",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(pady=(12, 2))
        ctk.CTkLabel(
            right_card,
            text="Mechanical walk  (all 4 angles used to estimate setup error)",
            font=ctk.CTkFont(size=10),
            text_color="gray",
        ).pack(pady=(0, 6))

        corr_tbl = ctk.CTkFrame(right_card, fg_color="transparent")
        corr_tbl.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self._build_result_table(corr_tbl, prefix="corr")

        # ── Bottom bar ────────────────────────────────────────────────────────
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", padx=20, pady=(4, 16))

        self._walk_label = ctk.CTkLabel(
            bottom,
            text=f"Walk Circle Radius: —   (Tolerance ≤ {TOLERANCE_MM:.1f} mm)",
            font=ctk.CTkFont(size=13),
        )
        self._walk_label.pack(side="left")

        self._report_btn = ctk.CTkButton(
            bottom,
            text="Generate Daily Report (PDF)",
            command=self._generate_report,
            width=210,
            state="disabled",
            font=ctk.CTkFont(size=13),
        )
        self._report_btn.pack(side="right")

    def _build_result_table(self, parent: ctk.CTkFrame, prefix: str):
        headers = ["Angle", "ΔX (mm)", "ΔY (mm)", "|ΔR| (mm)"]
        for c, h in enumerate(headers):
            parent.columnconfigure(c, weight=1)
            ctk.CTkLabel(
                parent,
                text=h,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#aaaaaa",
            ).grid(row=0, column=c, sticky="ew", padx=4, pady=2)

        sep = ctk.CTkFrame(parent, height=1, fg_color="#3a3a3a")
        sep.grid(row=1, column=0, columnspan=4, sticky="ew", padx=2, pady=3)

        for i, angle in enumerate(GANTRY_ANGLES):
            row = i + 2
            ctk.CTkLabel(
                parent,
                text=f"G{angle:03d}°",
                font=ctk.CTkFont(size=13, weight="bold"),
            ).grid(row=row, column=0, sticky="ew", padx=4, pady=7)

            for c, suffix in enumerate(["dx", "dy", "dr"], start=1):
                key = f"{prefix}_{angle}_{suffix}"
                lbl = ctk.CTkLabel(
                    parent,
                    text="—",
                    font=ctk.CTkFont(size=13, family="Courier"),
                )
                lbl.grid(row=row, column=c, sticky="ew", padx=4, pady=7)
                self._table_labels[key] = lbl

    # ── Event handlers ────────────────────────────────────────────────────────

    def _load_directory(self):
        directory = filedialog.askdirectory(title="Select DICOM Directory")
        if not directory:
            return

        self._dir_label.configure(text=Path(directory).name)
        self._pf_label.configure(text="Processing…", text_color="gray")
        self._pf_frame.configure(fg_color=["#2b2b2b", "#2b2b2b"])
        self.update()

        try:
            dcm_images      = load_dicom_images(directory)
            image_results   = {a: analyze_image(ds) for a, ds in dcm_images.items()}
            wl_results      = compute_wl_results(image_results)

            self._image_results   = image_results
            self._wl_results      = wl_results
            self._loaded_dir      = directory
            self._diag_fig_path   = generate_diagnostic_figure(image_results, wl_results)

            self._refresh_display()
            self._report_btn.configure(state="normal")

        except Exception as exc:
            messagebox.showerror("Processing Error", str(exc))
            self._pf_label.configure(text="— ERROR —", text_color="#f44336")

    def _refresh_display(self):
        if self._wl_results is None:
            return

        wl   = self._wl_results
        walk = wl["max_2d_walk_mm"]
        ok   = wl["pass_fail"]

        # PASS/FAIL banner
        if ok:
            self._pf_frame.configure(fg_color="#1a3d1e")
            self._pf_label.configure(
                text=f"PASS    {walk:.2f} mm",
                text_color="#66bb6a",
            )
        else:
            self._pf_frame.configure(fg_color="#3d1a1a")
            self._pf_label.configure(
                text=f"FAIL    {walk:.2f} mm",
                text_color="#ef5350",
            )

        self._walk_label.configure(
            text=(
                f"Walk Circle Radius: {walk:.3f} mm   "
                f"(Tolerance ≤ {TOLERANCE_MM:.1f} mm)"
            )
        )

        for angle in GANTRY_ANGLES:
            r = wl["per_angle"][angle]
            raw_dr  = np.sqrt(r["raw_dx"] ** 2 + r["raw_dy"] ** 2)
            corr_dr = np.sqrt(r["rel_dx"] ** 2 + r["rel_dy"] ** 2)

            def _color(v):
                if v < 0.5:
                    return "#66bb6a"
                if v < TOLERANCE_MM:
                    return "#ffa726"
                return "#ef5350"

            self._table_labels[f"raw_{angle}_dx"].configure(
                text=f"{r['raw_dx']:+.3f}", text_color=_color(abs(r["raw_dx"]))
            )
            self._table_labels[f"raw_{angle}_dy"].configure(
                text=f"{r['raw_dy']:+.3f}", text_color=_color(abs(r["raw_dy"]))
            )
            self._table_labels[f"raw_{angle}_dr"].configure(
                text=f"{raw_dr:.3f}", text_color=_color(raw_dr)
            )
            self._table_labels[f"corr_{angle}_dx"].configure(
                text=f"{r['rel_dx']:+.3f}", text_color=_color(abs(r["rel_dx"]))
            )
            self._table_labels[f"corr_{angle}_dy"].configure(
                text=f"{r['rel_dy']:+.3f}", text_color=_color(abs(r["rel_dy"]))
            )
            self._table_labels[f"corr_{angle}_dr"].configure(
                text=f"{corr_dr:.3f}", text_color=_color(corr_dr)
            )

    def _generate_report(self):
        if self._wl_results is None:
            messagebox.showwarning("No Data", "Load DICOM images first.")
            return

        default_name = f"WL_QA_{datetime.date.today().strftime('%Y%m%d')}.pdf"
        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=default_name,
            title="Save Daily QA Report",
        )
        if not save_path:
            return

        try:
            generate_pdf_report(
                self._wl_results,
                self._image_results,
                save_path,
                diag_fig_path=self._diag_fig_path,
            )
            messagebox.showinfo("Report Saved", f"Report saved to:\n{save_path}")
        except Exception as exc:
            messagebox.showerror("Report Error", str(exc))


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
    story.append(Paragraph(f"{MACHINE_NAME}  —  {PHANTOM_NAME}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=_BORDER, spaceAfter=8))

    # ── Metadata table ────────────────────────────────────────────────────────
    today_str = datetime.date.today().strftime("%B %d, %Y")
    time_str  = datetime.datetime.now().strftime("%H:%M")
    max_walk  = wl_results["max_2d_walk_mm"]
    passed    = wl_results["pass_fail"]
    baseline_dx = wl_results["baseline_dx"]
    baseline_dy = wl_results["baseline_dy"]
    baseline_dr = np.sqrt(baseline_dx ** 2 + baseline_dy ** 2)

    meta_rows = [
        ["Date:", today_str,        "Time:",        time_str],
        ["Machine:", MACHINE_NAME,  "Phantom:",     PHANTOM_NAME],
        ["Field Size:", f"{FIELD_SIZE_MM:.0f}×{FIELD_SIZE_MM:.0f} mm",
         "Void Diameter:", f"{VOID_DIAMETER_MM:.1f} mm (air)"],
        ["Tolerance:", f"≤ {TOLERANCE_MM:.1f} mm",
         "MEC Setup Error:", f"{baseline_dr:.3f} mm  (ΔX={baseline_dx:+.3f}, ΔY={baseline_dy:+.3f})"],
    ]
    meta_tbl = Table(meta_rows, colWidths=[1.1*inch, 2.1*inch, 1.3*inch, 2.5*inch])
    meta_tbl.setStyle(TableStyle([
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("FONTNAME",   (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",   (2, 0), (2, -1),  "Helvetica-Bold"),
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
        f"MEC centre (setup error estimate): "
        f"ΔX={baseline_dx:+.3f} mm, ΔY={baseline_dy:+.3f} mm  |  "
        f"Walk circle radius: {max_walk:.3f} mm",
        ParagraphStyle("mecnote", parent=ss["Normal"], fontSize=9,
                       textColor=colors.HexColor("#444444"), spaceAfter=4),
    ))

    raw_header = [
        "Gantry", "ΔX raw (mm)", "ΔY raw (mm)", "|ΔR| raw (mm)", "Note"
    ]
    raw_data = [raw_header]
    for angle in GANTRY_ANGLES:
        r  = wl_results["per_angle"][angle]
        dr = np.sqrt(r["raw_dx"] ** 2 + r["raw_dy"] ** 2)
        note = ""
        raw_data.append([
            f"G{angle:03d}°",
            f"{r['raw_dx']:+.3f}",
            f"{r['raw_dy']:+.3f}",
            f"{dr:.3f}",
            note,
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
        "Corrected Displacements  (MEC Setup Error Removed — Mechanical Isocenter Walk)",
        h2_style,
    ))

    corr_header = [
        "Gantry", "ΔX corr (mm)", "ΔY corr (mm)", "|ΔR| corr (mm)", "Status"
    ]
    corr_data = [corr_header]
    for angle in GANTRY_ANGLES:
        r  = wl_results["per_angle"][angle]
        dr = np.sqrt(r["rel_dx"] ** 2 + r["rel_dy"] ** 2)
        if angle == 0:
            status_cell = "—  (reference)"
        else:
            status_cell = "✓  PASS" if dr <= TOLERANCE_MM else "✗  FAIL"
        corr_data.append([
            f"G{angle:03d}°",
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
    corr_tbl = Table(corr_data, colWidths=[0.9*inch, 1.2*inch, 1.2*inch, 1.3*inch, 2.5*inch])
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
        "<b>Walk Circle (Minimum Enclosing Circle):</b>  The residual CBCT setup error is estimated "
        "from ALL four gantry angles as the centre of the Minimum Enclosing Circle (MEC) of the four "
        "raw void-to-field-centre displacement vectors (ΔX, ΔY). "
        "Using all four images avoids the coordinate-system ambiguity of single-angle baseline "
        "subtraction and gives a geometry-independent estimate of the static setup error. "
        "Subtracting the MEC centre from each raw vector yields the per-angle mechanical walk. "
        "The MEC radius is the smallest circle containing all four walk vectors — the walk circle metric.",
        note_style,
    ))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(
        f"<b>Tolerance:</b>  Walk circle radius ≤ {TOLERANCE_MM:.1f} mm = PASS.  "
        "Pixel spacing corrected from detector plane to isocenter using "
        "SAD/SID magnification (spacing_iso = spacing_det × SAD/SID).",
        note_style,
    ))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.18 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceAfter=4))
    story.append(Paragraph(
        f"Generated by Winston-Lutz QA Tool v1.0  —  "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  —  "
        f"{MACHINE_NAME}  /  {PHANTOM_NAME}",
        footer_style,
    ))

    doc.build(story)
    print(f"PDF report written to: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = WLApp()
    app.mainloop()
