"""Native Elekta iView export: `img_<uid>/` folders holding a lossless-JPEG image and a `_Frame.xml` header (no DICOM, no patient id, no pixel spacing).

* decode(): the lossless JPEG (SOF3, single component, Huffman), pure numpy / python (about 1 s per image).
* NativeDS: an object that looks like the DICOM dataset the rest of the app reads (GantryAngle, RTImageLabel, PatientID, dates, SOPInstanceUID, pixel_array,
  spacing tags), so `analyze_image` / `analyze_picket_fence` take it unchanged. The panel's physical pitch is 0.4 mm at SID 1600 mm (0.25 mm at isocenter);
  the MOSAIQ DICOM export of the same panel says 0.405 mm.
* scan_native(): the img_ folders under a folder, classified from the image itself (Daily WL BB image / picket fence / other), cached by file time and size.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

PIXEL_PITCH_MM = 0.4            # at the panel (SID 1600)
SAD_MM, SID_MM = 1000.0, 1600.0
CACHE_NAME = "native_scan_cache.json"
_UID_RE = re.compile(r"\.(\d{9,10})\.(\d+)$")



def _parse(d):
    pos, hts, dri = 2, {}, 0
    sof = None
    while pos < len(d):
        if d[pos] != 0xFF:
            raise ValueError("bad marker")
        m = d[pos + 1]
        L = int.from_bytes(d[pos + 2:pos + 4], "big")
        seg = d[pos + 4:pos + 2 + L]
        if m == 0xC3:
            sof = (seg[0], int.from_bytes(seg[1:3], "big"), int.from_bytes(seg[3:5], "big"), seg[5])
        elif m == 0xC4:
            p = 0
            while p < len(seg):
                tc_th = seg[p]
                counts = seg[p + 1:p + 17]
                n = sum(counts)
                hts[tc_th & 0x0F] = (counts, seg[p + 17:p + 17 + n])
                p += 17 + n
        elif m == 0xDD:
            dri = int.from_bytes(seg[:2], "big")
        elif m == 0xDA:
            ncomp = seg[0]
            tbl = seg[2] >> 4 if ncomp == 1 else 0
            ss, al = seg[1 + 2 * ncomp], seg[3 + 2 * ncomp] & 0x0F
            return sof, hts[tbl], dri, ss, al, pos + 2 + L
        elif m in (0xC0, 0xC1, 0xC2):
            raise ValueError("not lossless JPEG")
        pos += 2 + L
    raise ValueError("no SOS")


def decode(path):
    d = open(path, "rb").read()
    (prec, h, w, nc), (counts, vals), dri, ss, al, start = _parse(d)
    if nc != 1:
        raise ValueError("only single component supported")
    # Unstuff FF00 -> FF, split at restart markers.
    raw = d[start:]
    if raw.endswith(b"\xff\xd9"):
        raw = raw[:-2]
    # Canonical Huffman -> {(length, code): symbol}
    table, code, k = {}, 0, 0
    for ln in range(1, 17):
        for _ in range(counts[ln - 1]):
            table[(ln, code)] = vals[k]
            code += 1
            k += 1
        code <<= 1
    out = np.zeros(h * w, dtype=np.int32)
    segs = _split_restarts(raw) if dri else [raw.replace(b"\xff\x00", b"\xff")]
    if not dri:
        dri = h * w
    idx = 0
    half = 1 << (prec - al - 1)
    for sg in segs:
        n = min(dri, h * w - idx)
        idx = _decode_segment(sg, n, table, out, idx, w, ss, half, prec)
    img = (out.reshape(h, w) << al).astype(np.int64)
    return (img & ((1 << prec) - 1)).astype(np.uint16)


def _split_restarts(raw):
    parts, cur, i = [], bytearray(), 0
    while i < len(raw):
        b = raw[i]
        if b == 0xFF and i + 1 < len(raw):
            n = raw[i + 1]
            if n == 0x00:
                cur.append(0xFF); i += 2; continue
            if 0xD0 <= n <= 0xD7:
                parts.append(bytes(cur)); cur = bytearray(); i += 2; continue
        cur.append(b); i += 1
    parts.append(bytes(cur))
    return parts


def _decode_segment(data, n, table, out, idx, w, ss, half, prec):
    # 16-bit prefix lookup: lut[peek16] = (code length, symbol)
    lut = _lut(table)
    buf = data + b"\x00" * 8
    acc, nbits, p = 0, 0, 0
    vals = [0] * n
    mask = (1 << 16) - 1
    for k in range(n):
        while nbits < 32:
            acc = (acc << 8) | buf[p]; p += 1; nbits += 8
        ln, s = lut[(acc >> (nbits - 16)) & mask]
        if ln == 0:
            raise ValueError("bad huffman code")
        nbits -= ln
        if s == 0:
            diff = 0
        elif s == 16:
            diff = 32768
        else:
            v = (acc >> (nbits - s)) & ((1 << s) - 1)
            nbits -= s
            diff = v if v >= (1 << (s - 1)) else v - (1 << s) + 1
        acc &= (1 << nbits) - 1
        y, x = divmod(k, w)
        if k == 0:
            pred = half
        elif y == 0:
            pred = vals[k - 1]
        elif x == 0:
            pred = vals[k - w]
        else:
            a, b, c = vals[k - 1], vals[k - w], vals[k - w - 1]
            if ss == 1: pred = a
            elif ss == 2: pred = b
            elif ss == 3: pred = c
            elif ss == 4: pred = a + b - c
            elif ss == 5: pred = a + ((b - c) >> 1)
            elif ss == 6: pred = b + ((a - c) >> 1)
            else: pred = (a + b) >> 1
        vals[k] = (pred + diff) & 0xFFFF
    out[idx:idx + n] = vals
    return idx + n


def _lut(table):
    lut = [(0, 0)] * 65536
    for (ln, code), sym in table.items():
        lo = code << (16 - ln)
        for i in range(lo, lo + (1 << (16 - ln))):
            lut[i] = (ln, sym)
    return lut




# ── header, adapter, classification ───────────────────────────────────────────────────────────────────────────

def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_frame_xml(folder: Path) -> dict:
    try:
        root = ET.parse(Path(folder) / "_Frame.xml").getroot()
    except (OSError, ET.ParseError):
        return {}
    g = _num(root.findtext("Frames/Frame/gantryAngle"))
    return {"energy": (root.findtext("Image/Energy") or "").strip(), "mu": _num(root.findtext("Image/TotalSetBeamMU")),
            "gantry": None if g is None else g % 360.0, "uid": (root.findtext("Image/DicomUID") or "").strip()}


def folder_time(name: str, jpg: Path):
    """(acquisition time, image number): the epoch in the folder name, else the file time."""
    m = _UID_RE.search(name)
    if m and 1.0e9 < int(m.group(1)) < 4.0e9:
        return dt.datetime.fromtimestamp(int(m.group(1))), int(m.group(2))
    return dt.datetime.fromtimestamp(jpg.stat().st_mtime), None


def find_jpg(folder: Path):
    try:
        return next((p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg")), None)
    except OSError:
        return None


class NativeDS:
    """Duck-typed stand-in for a pydicom RTIMAGE dataset."""

    def __init__(self, folder: Path, label: str = "", patient: str = ""):
        self.folder = Path(folder)
        jpg = find_jpg(self.folder)
        if jpg is None:
            raise ValueError(f"no image in {folder}")
        h = read_frame_xml(self.folder)
        when, seq = folder_time(self.folder.name, jpg)
        self._jpg = jpg
        self._arr = None
        self.Modality = "RTIMAGE"
        self.RTImageLabel = label
        self.SeriesDescription = label
        self.PatientID = patient
        self.SOPInstanceUID = h.get("uid") or self.folder.name[4:]
        self.GantryAngle = h.get("gantry") if h.get("gantry") is not None else 0.0
        self.StudyDate = self.ContentDate = f"{when:%Y%m%d}"
        self.StudyTime = self.ContentTime = f"{when:%H%M%S}"
        self.when, self.seq, self.mu, self.energy = when, seq, h.get("mu"), h.get("energy", "")
        self.ImagePlanePixelSpacing = [PIXEL_PITCH_MM, PIXEL_PITCH_MM]
        self.RadiationMachineSAD, self.RTImageSID = SAD_MM, SID_MM
        self.RescaleSlope, self.RescaleIntercept = 1.0, 0.0
        self.is_native = True

    @property
    def pixel_array(self) -> np.ndarray:
        if self._arr is None:
            self._arr = decode(self._jpg)
        return self._arr

    @property
    def Rows(self):
        return self.pixel_array.shape[0]

    @property
    def Columns(self):
        return self.pixel_array.shape[1]


def measure(arr: np.ndarray, mm: float = PIXEL_PITCH_MM * SAD_MM / SID_MM) -> dict:
    """Shape of the beam in one image (mm at isocenter) and its kind: 'wl' (40 x 40 mm field with the phantom's void), 'pf' (the long single picket) or 'other'."""
    a = ndi.uniform_filter(arr.astype(float), 4)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    mid = (lo + hi) / 2
    beam = a < mid if np.median(a) > mid else a > mid           # iView stores the beam dark
    lab, n = ndi.label(beam)
    if n == 0:
        return {"kind": "other"}
    sizes = ndi.sum(beam, lab, range(1, n + 1))
    m = lab == 1 + int(np.argmax(sizes))
    n_big = int((sizes > 0.3 * sizes.max()).sum())
    ys, xs = np.nonzero(m)
    ev, _ = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
    length, width = 4 * np.sqrt(max(ev[1], 0)) * mm, 4 * np.sqrt(max(ev[0], 1e-9)) * mm
    bb = ((ys.max() - ys.min() + 1) * mm, (xs.max() - xs.min() + 1) * mm)
    er = ndi.binary_erosion(m, iterations=max(int(5 / mm), 1))
    cv = float(arr[er].std() / arr[er].mean()) if er.sum() > 100 else 0.0
    f = {"length": float(length), "width": float(width), "bbox_y": float(bb[0]), "bbox_x": float(bb[1]), "cv": cv, "strips": n_big}
    long_, short_ = max(bb), min(bb)
    if long_ >= 170 and short_ <= 30:
        f["kind"] = "pf"                                         # the Daily picket fence: one 1 cm gap, 20 cm long (collimator slits are 11-14 cm long)
    elif 36 <= long_ <= 46 and 36 <= short_ <= 46 and long_ / max(short_, 1e-6) < 1.15 and cv >= 0.015:
        f["kind"] = "wl"                                         # the Daily WL field is 40 x 40 mm with the phantom's void (the monthly BB field is 30 x 30 mm)
    else:
        f["kind"] = "other"
    return f


def is_native_folder(folder) -> bool:
    folder = Path(folder)
    try:
        return any(p.is_dir() and p.name.lower().startswith("img_") for p in folder.rglob("*") if p.is_dir()) or folder.name.lower().startswith("img_")
    except OSError:
        return False


def scan_native(folder, cache_dir=None, progress=None) -> list[NativeDS]:
    """Every img_ folder under `folder`, oldest first, each labelled 'WL' / 'PF' / '' from the image (cached when cache_dir is given)."""
    folder = Path(folder)
    dirs = sorted(p for p in ([folder] if folder.name.lower().startswith("img_") else folder.rglob("*")) if p.is_dir() and p.name.lower().startswith("img_"))
    cache_file = Path(cache_dir) / CACHE_NAME if cache_dir else None
    try:
        cache = json.loads(cache_file.read_text()) if cache_file else {}
    except (OSError, ValueError):
        cache = {}
    fresh, out = {}, []
    for i, d in enumerate(dirs):
        if progress:
            progress(i, len(dirs))
        jpg = find_jpg(d)
        if jpg is None:
            continue
        st = jpg.stat()
        c = cache.get(str(d))
        if c and c.get("m") == st.st_mtime and c.get("s") == st.st_size:
            kind = c["kind"]
        else:
            try:
                kind = measure(decode(jpg))["kind"]
            except Exception:
                kind = "other"
        fresh[str(d)] = {"m": st.st_mtime, "s": st.st_size, "kind": kind}
        try:
            ds = NativeDS(d, label={"wl": "WL", "pf": "PF"}.get(kind, ""))
        except ValueError:
            continue
        ds.kind = kind
        out.append(ds)
    if cache_file:
        try:
            cache_file.write_text(json.dumps(fresh))
        except OSError:
            pass
    out.sort(key=lambda d: (d.when, d.seq or 0))
    return out
