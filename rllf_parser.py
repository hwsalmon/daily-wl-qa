"""Parser for RadCalc ".rllf" MLC log files.

RadCalc (LAP/RadCalc QA) wraps a capture of Elekta's own EOS/RTD real-time
telemetry stream (its header literally contains the marker
``ELEKTA_EOS_RTD_VX2``) inside a proprietary ``LSIRLL1`` container. This is a
*different* on-disk format from the native Elekta binary TRF servo log
(`TRFParser` in `logfile_engine.py`), but samples the same underlying
per-axis actual/commanded MLC data, just at ~1 Hz instead of TRF's 25 Hz and
as ASCII decimal text instead of packed int16.

`RLLFParser` exposes the same public surface as `TRFParser` (`time_s`,
`cp_index`, `gantry_actual`, `bank_a_actual`/`bank_b_actual`,
`bank_a_expect`/`bank_b_expect`, `beam_on_mask`, `is_static`, `timestamp`,
`patient_id`, `plan_name`) so it can be used as a drop-in replacement
anywhere a `TRFParser` is used -- in particular by the unmodified
`MLCQAAnalyzer` in `logfile_engine.py`.

## Container format

``LSIRLL1`` magic, then a sequence of per-second frames. Each frame starts
with a 34-byte preamble (a monotonic frame counter and a millisecond
timestamp that increments by exactly 1000 between frames -- i.e. 1 Hz)
followed by a flat run of TLV records:

    [4-byte big-endian tag][1-byte type][4-byte big-endian length][ASCII data]

The type byte is 'P' (planned -- almost always blank for MLC leaves, this
format has no per-leaf planned value), 'R' (read/actual, machine-measured),
or 'S' (setting/commanded target) -- directly analogous to TRF's
actual-vs-expect pair. Numeric values are ASCII decimal **centimetres**
(TRF's raw int16 counts are millimetres/10; these are already text, just in
a different unit), so must be multiplied by 10 for mm.

## MLC leaf tag layout and the bank remap (empirically derived)

Leaf position tags are `0x50010101`-`0x50010150` ("device bank 1", 80
leaves) and `0x50010201`-`0x50010250` ("device bank 2", 80 leaves), each
with 'R' and 'S' sub-records. Gantry angle is `0x50010007` and collimator
angle `0x50010008` (both verified directly against this session's filenames,
e.g. "Gantry 090 Coll 180").

The device banks do **not** correspond 1:1 to DICOM bank_a/bank_b in tag
order and sign -- confirmed by aligning against the reference RT Plan
(`Log files/QAMLCIN2020_MLCQA2020.dcm`, the same plan already used for the
TRF-based log QA) across its three distinct picket-fence hold positions:

    device bank 1, tag order i=0..79  ==  DICOM bank_b, reversed (i -> 79-i), same sign
    device bank 2, tag order i=0..79  ==  DICOM bank_a, reversed (i -> 79-i), NEGATED sign

(mean abs error ~0.85 mm across all 80 leaves x 3 plateaus with this
mapping, consistent with real leaf noise, vs. several mm/cm with any other
sign/order/bank combination.) So, mirroring the way `MLCQAAnalyzer` expects
`TRFParser.bank_a_actual`/`bank_b_actual` to look *before* its own
`[:, ::-1]` un-reversal:

    bank_a_actual = -device_bank_2   (already leaf-order-compatible, no extra reversal)
    bank_b_actual =  device_bank_1

If a different RadCalc export (different plan, different site) is ever
decoded, re-derive this mapping the same way rather than assuming it
carries over -- align its distinct hold positions against a known
reference plan, exactly as `PANEL_OFFSET_LEAVES` was derived for the
picket-fence image QA (see CLAUDE.md).

## Control-point index

Unlike TRF, this format has no explicit per-frame control-point/segment
index axis. Instead, `cp_index` is *derived* here from the commanded ('S')
leaf vector itself: whenever it changes by more than `_CP_CHANGE_THRESH_MM`
from the previous frame, a new segment has begun. The reference plan used
by this app always has each real control point duplicated as two identical
consecutive DICOM control points (CP0==CP1, CP2==CP3, CP4==CP5 -- three
real picket holds), and Elekta's own segment numbering (as read directly
off real TRF logs of the exact same plan) numbers each hold's *odd*
DICOM control point (1, 3, 5, ...) as its segment id. So the j-th detected
hold here is assigned `cp_index = 2*j + 1` to match that existing, already
plan-matched convention used by `MLCQAAnalyzer` -- not an arbitrary choice.
"""
from __future__ import annotations

import os
import re
import struct

import numpy as np

N_LEAVES_PER_BANK = 80
_TAG_GANTRY = "50010007"
_TAG_COLLIMATOR = "50010008"
_TAG_PLAN_NAME = "70010002"
_TAG_SITE = "70010005"
_BANK1_TAGS = [f"5001{0x0100 + i:04x}" for i in range(1, N_LEAVES_PER_BANK + 1)]
_BANK2_TAGS = [f"5001{0x0200 + i:04x}" for i in range(1, N_LEAVES_PER_BANK + 1)]

_CM_TO_MM = 10.0
_CP_CHANGE_THRESH_MM = 2.0  # real hold-to-hold leaf moves are cm-scale; 'S' jitter is <0.1mm

_FILENAME_TS_RE = re.compile(r"_(\d{14})\.rllf$", re.IGNORECASE)

_TYPE_BYTES = (0x50, 0x70, 0x52, 0x72, 0x53, 0x73)


def _try_parse_record(data: bytes, p: int):
    if p + 9 > len(data):
        return None
    tb = data[p + 4]
    if tb not in _TYPE_BYTES:
        return None
    length = struct.unpack_from(">I", data, p + 5)[0]
    if length > 300:
        return None
    start = p + 9
    end = start + length
    if end > len(data):
        return None
    chunk = data[start:end]
    if length > 0 and not all(32 <= c < 127 for c in chunk):
        return None
    # 'P'/'R'/'S' with bit 0x20 set (0x70/0x72/0x73) vs clear (0x50/0x52/0x53) are the
    # same logical type -- the bit distinguishes something else in the container we
    # don't need (empirically doesn't correlate with anything QA-relevant).
    type_char = chr(tb & 0xDF)
    return data[p:p + 4].hex(), type_char, chunk.decode("ascii").strip(), end


def _parse_frames(data: bytes) -> list[list[tuple[str, str, str]]]:
    """Split the record stream into per-frame lists of (tag, type, value)."""
    n = len(data)
    magic_len = struct.unpack_from(">I", data, 0)[0]
    pos = 4 + magic_len
    if data[4:pos].decode("ascii", errors="replace") != "LSIRLL1":
        raise ValueError(f"Not an LSIRLL1 .rllf file: bad magic at start of {data[:20]!r}")

    # A fixed-format ~33-byte per-frame preamble (frame counter + a millisecond
    # timestamp that increments by exactly 1000 between frames) separates each
    # frame's record run from the next. It carries no QA-relevant data, so it's
    # simplest to just treat it as an unparseable "gap" and skip over it, the
    # same way inter-record gaps are skipped below.
    frames: list[list[tuple[str, str, str]]] = []
    cur: list[tuple[str, str, str]] = []
    p = pos
    while p < n:
        rec = _try_parse_record(data, p)
        if rec is None:
            if cur:
                frames.append(cur)
            cur = []
            while p < n and _try_parse_record(data, p) is None:
                p += 1
            continue
        tag, type_char, value, end = rec
        cur.append((tag, type_char, value))
        p = end
    if cur:
        frames.append(cur)
    return frames


class RLLFParser:
    """Parse a RadCalc `.rllf` MLC log file into the same shape as `TRFParser`."""

    def __init__(self, path: str):
        self.path = path
        self._parse()

    def _parse(self):
        with open(self.path, "rb") as f:
            data = f.read()
        frames = _parse_frames(data)
        self.n_frames = len(frames)
        if self.n_frames == 0:
            raise ValueError(f"No frames found in RLLF file: {self.path}")

        # Every value (leaf positions, gantry, jaws, ...) is delta/change-encoded:
        # a tag is only re-emitted within a frame when its value actually changes
        # (plus an occasional keepalive), not on every frame. So state must be
        # carried forward across frames -- each frame's effective dict is "last
        # known value of every tag seen so far", updated by whatever that frame's
        # own records changed, not just the reads that happen to fall in it.
        frame_dicts = []
        current: dict[str, dict[str, str]] = {}
        for fr in frames:
            for tag, type_char, value in fr:
                current.setdefault(tag, {})[type_char] = value
            frame_dicts.append({tag: dict(sub) for tag, sub in current.items()})

        def _floats(tag: str, type_char: str) -> np.ndarray:
            out = np.full(self.n_frames, np.nan)
            for i, d in enumerate(frame_dicts):
                v = d.get(tag, {}).get(type_char, "")
                if v:
                    try:
                        out[i] = float(v)
                    except ValueError:
                        pass
            return out

        def _leaf_matrix(tags: list[str], type_char: str) -> np.ndarray:
            out = np.full((self.n_frames, len(tags)), np.nan)
            for i, d in enumerate(frame_dicts):
                for j, tag in enumerate(tags):
                    v = d.get(tag, {}).get(type_char, "")
                    if v:
                        try:
                            out[i, j] = float(v)
                        except ValueError:
                            pass
            return out * _CM_TO_MM

        self.gantry_actual = _floats(_TAG_GANTRY, "R")
        self.collimator_actual = _floats(_TAG_COLLIMATOR, "R")

        device_bank1_actual = _leaf_matrix(_BANK1_TAGS, "R")
        device_bank2_actual = _leaf_matrix(_BANK2_TAGS, "R")
        device_bank1_expect = _leaf_matrix(_BANK1_TAGS, "S")
        device_bank2_expect = _leaf_matrix(_BANK2_TAGS, "S")

        # See module docstring: empirically-derived bank/sign/order remap.
        self.bank_a_actual = -device_bank2_actual
        self.bank_b_actual = device_bank1_actual
        self.bank_a_expect = -device_bank2_expect
        self.bank_b_expect = device_bank1_expect

        self._compute_cp_index(device_bank1_expect, device_bank2_expect)

        # Frame timing: derive from the 1 Hz frame counter (index itself is enough
        # since spacing is a constant 1000 ms, verified against the preamble).
        self.time_s = np.arange(self.n_frames, dtype=float)

        plan_name = frame_dicts[0].get(_TAG_PLAN_NAME, {}).get("P", "").strip()
        self.plan_name = plan_name or os.path.splitext(os.path.basename(self.path))[0]
        self.patient_id = frame_dicts[0].get(_TAG_SITE, {}).get("P", "").strip() or "unknown"

        m = _FILENAME_TS_RE.search(os.path.basename(self.path))
        if m:
            ts = m.group(1)
            self.timestamp = f"{ts[2:4]}/{ts[4:6]}/{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]} Z"
        else:
            self.timestamp = ""

    def _compute_cp_index(self, bank1_expect: np.ndarray, bank2_expect: np.ndarray):
        """Derive a TRF-style 1-based-odd control-point index from commanded-position
        changes (see module docstring for why odd indices 1, 3, 5, ...)."""
        combined = np.concatenate([bank1_expect, bank2_expect], axis=1)  # (n_frames, 160)
        cp_index = np.zeros(self.n_frames, dtype=float)
        segment = 0
        prev = combined[0]
        cp_index[0] = 2 * segment + 1
        for i in range(1, self.n_frames):
            cur = combined[i]
            diff = np.nanmax(np.abs(cur - prev)) if not np.all(np.isnan(cur - prev)) else 0.0
            if np.isfinite(diff) and diff > _CP_CHANGE_THRESH_MM:
                segment += 1
            cp_index[i] = 2 * segment + 1
            prev = cur
        self.cp_index = cp_index
        self.n_segments = segment + 1

    @property
    def beam_on_mask(self) -> np.ndarray:
        """RLLF only captures frames while the beam is delivering, so every parsed
        frame is beam-on."""
        return np.ones(self.n_frames, dtype=bool)

    @property
    def is_static(self) -> bool:
        g = self.gantry_actual[~np.isnan(self.gantry_actual)]
        if g.size < 2:
            return True
        return float(np.ptp(g)) < 2.0

    @property
    def coll_angle_from_name(self) -> float | None:
        m = re.search(r"[Cc]oll\s*(\d+)", self.plan_name)
        return float(m.group(1)) if m else None
