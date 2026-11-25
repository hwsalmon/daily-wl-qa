"""Read new Daily WL sessions straight from a machine's EPID folder, and (optionally) pair every image with its machine log.

* scan_epid_folder(): recursively reads the DICOM *headers* under the folder (cached, so a rescan of thousands of files is quick), picks the Daily WL images
  (RTImageLabel "WL1".."WL4"), groups them into sessions (one machine, no gap of 45 min or more) and reports which cardinal gantry angles each session has.
* stage_session(): copies a session's four WL images (and that day's Picket Fence image, if any) into a temporary folder, so the normal
  "Load DICOM Directory" path processes it exactly like a manual import.
* LogIndex / attach_logs(): optional. Reads the machine's .rllf delivery logs and pairs each WL image with the log of the delivery that made it, giving the aperture
  (MLC edges, jaws) the collimator actually had. Without a log folder, or when nothing matches, everything works as before.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom

CARDINALS = (0, 90, 180, 270)
CARDINAL_TOL_DEG = 5.0
SESSION_GAP_S = 45 * 60
CACHE_NAME = "epid_scan_cache.json"


# ── DICOM header scan ─────────────────────────────────────────────────────────────────────────────────────────

def _when(ds) -> dt.datetime | None:
    for dtag, ttag in (("ContentDate", "ContentTime"), ("StudyDate", "StudyTime"), ("AcquisitionDate", "AcquisitionTime")):
        d, t = str(getattr(ds, dtag, "") or ""), str(getattr(ds, ttag, "") or "")
        if len(d) == 8 and t:
            try:
                return dt.datetime.strptime(d + t.split(".")[0].zfill(6)[:6], "%Y%m%d%H%M%S")
            except ValueError:
                continue
    return None


def read_header(path: Path) -> dict | None:
    """The fields the scan needs, or None for a file that is not an EPID RTIMAGE."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        if not getattr(ds, "SOPInstanceUID", None) or (getattr(ds, "Modality", "") != "RTIMAGE" and not getattr(ds, "RTImageLabel", None)):
            return None
        when = _when(ds)
        g = getattr(ds, "GantryAngle", None)
        return {"sop": str(ds.SOPInstanceUID), "patient": str(getattr(ds, "PatientID", "") or ""), "label": str(getattr(ds, "RTImageLabel", "") or ""),
                "series": str(getattr(ds, "SeriesDescription", "") or ""), "gantry": None if g in (None, "") else float(g),
                "when": when.isoformat() if when else ""}
    except Exception:
        return None


def is_pf(h: dict) -> bool:
    t = (h["label"] + " " + h["series"]).upper()
    return "PF" in t or "PICKET" in t


def is_wl(h: dict) -> bool:
    return not is_pf(h) and bool(re.match(r"\s*WL\d*", h["label"], re.I) or re.match(r"\s*WL\d*", h["series"], re.I))


def _cardinal(g: float | None) -> int | None:
    if g is None:
        return None
    best = min(CARDINALS, key=lambda a: min(abs(a - g), abs(360 + a - g), abs(a - g - 360)))
    diff = min(abs(best - g), abs(360 + best - g), abs(best - g - 360))
    return best if diff <= CARDINAL_TOL_DEG else None


@dataclass
class Session:
    patient: str
    when: dt.datetime
    files: dict = field(default_factory=dict)        # cardinal angle -> path of the latest image at that angle
    sops: dict = field(default_factory=dict)         # cardinal angle -> SOPInstanceUID
    pf: Path | None = None
    machine: str = ""
    status: str = "new"                              # new | processed | skipped
    preview: dict | None = None                      # filled by the caller: {"displacement": .., "pass": ..}

    @property
    def missing(self) -> list:
        return [a for a in CARDINALS if a not in self.files]

    @property
    def complete(self) -> bool:
        return not self.missing

    @property
    def key(self) -> str:
        return f"{self.patient}|{self.when:%Y%m%d%H%M%S}"


STATE_NAME = "epid_scan_state.json"
SAFETY_S = 3 * 86400            # folders changed within 3 days before the last mark are looked at again (late images, clock differences)


def _state(cache_dir) -> dict:
    try:
        return json.loads((Path(cache_dir) / STATE_NAME).read_text())
    except (OSError, ValueError):
        return {}


def scan_mark(folder, cache_dir) -> float | None:
    """The time before which everything in `folder` has been handled (None = never scanned, or the scan must be complete)."""
    return (_state(cache_dir).get(str(folder)) or {}).get("mark")


def save_scan_mark(folder, cache_dir, sessions, newest: float, since_ok: bool = True) -> None:
    """Remember how far this folder has been dealt with: the newest file seen, but never beyond the oldest session still waiting (new or incomplete),
    so what is not yet imported is offered again next time. Everything older is skipped by the next scan."""
    mark = newest
    for x in sessions:
        if x.status == "new":
            for p in list(x.files.values()) + ([x.pf] if x.pf else []):
                try:
                    mark = min(mark, Path(p).stat().st_mtime)
                except OSError:
                    mark = min(mark, 0.0)
    st = _state(cache_dir)
    st[str(folder)] = {"mark": mark, "saved": dt.datetime.now().isoformat(timespec="seconds")}
    try:
        (Path(cache_dir) / STATE_NAME).write_text(json.dumps(st))
    except OSError:
        pass


def scan_epid_folder(folder: str | Path, cache_dir: Path, progress=None, max_files: int = 200000, since: float | None = None) -> list[Session]:
    """All Daily WL sessions found under `folder` (newest first). `progress(i, n)` is called while headers are read.
    `since` (see scan_mark): look only at folders and files changed at or after `since` - the rest was imported already; None reads everything.
    The newest file time seen is left in `scan_epid_folder.newest` for save_scan_mark."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"EPID folder not found: {folder}")
    import iview_native
    cut = None if since is None else since - SAFETY_S
    tree = iview_native.walk_tree(folder, cut)
    scan_epid_folder.newest = tree[2]
    cache_file = Path(cache_dir) / CACHE_NAME
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        cache = {}
    files = [(p, m, sz) for p, m, sz in tree[0] if p.suffix.lower() not in (".jpg", ".jpeg", ".xml", ".rllf")][:max_files]
    heads: list[tuple[Path, dict]] = []
    fresh = {}
    for i, (p, mt, sz) in enumerate(files):
        if progress and i % 25 == 0:
            progress(i, len(files))
        key = str(p)
        c = cache.get(key)
        if c and c.get("m") == mt and c.get("s") == sz:
            h = c.get("h")
        else:
            h = read_header(p) if sz > 10_000 else None
        fresh[key] = {"m": mt, "s": sz, "h": h}
        if h:
            heads.append((p, h))
    try:
        cache.update(fresh)                                       # earlier, older files stay known
        cache_file.write_text(json.dumps(cache))
    except OSError:
        pass
    try:                                                           # native iView exports (img_<uid> folders): classified from the image, no patient id
        import iview_native
        for d in iview_native.scan_native(folder, cache_dir, since=cut, tree=tree):
            if d.kind in ("wl", "pf"):
                heads.append((d.folder, {"sop": d.SOPInstanceUID, "patient": "", "label": "WL" if d.kind == "wl" else "PF", "series": "",
                                         "gantry": float(d.GantryAngle), "when": d.when.isoformat()}))
    except Exception:
        pass
    wl = sorted(((p, h) for p, h in heads if is_wl(h) and h["when"]), key=lambda x: (x[1]["patient"], x[1]["when"]))
    pfs = [(p, h) for p, h in heads if is_pf(h) and h["when"]]
    sessions: list[Session] = []
    for p, h in wl:
        when = dt.datetime.fromisoformat(h["when"])
        cur = sessions[-1] if sessions else None
        if cur is None or cur.patient != h["patient"] or (when - cur._last).total_seconds() >= SESSION_GAP_S:
            cur = Session(patient=h["patient"], when=when)
            cur._first = cur._last = when
            sessions.append(cur)
        cur._last = when
        a = _cardinal(h["gantry"])
        if a is None:
            continue
        if a not in cur.files or when >= cur.sop_when.get(a, dt.datetime.min):          # a repeated delivery replaces the earlier one
            cur.files[a], cur.sops[a] = p, h["sop"]
            cur.sop_when[a] = when
    for s in sessions:
        same_day = [(p, h) for p, h in pfs if h["patient"] == s.patient and dt.datetime.fromisoformat(h["when"]).date() == s.when.date()]
        if same_day:
            s.pf = min(same_day, key=lambda x: abs((dt.datetime.fromisoformat(x[1]["when"]) - s.when).total_seconds()))[0]
    return sorted(sessions, key=lambda s: s.when, reverse=True)


# Session needs a couple of private attributes while it is built
Session.sop_when = property(lambda self: self.__dict__.setdefault("_sop_when", {}))


def stage_session(s: Session) -> Path:
    """Copy the session's images into a fresh temporary folder (the normal directory-load path then handles them)."""
    tmp = Path(tempfile.mkdtemp(prefix="wl_epid_"))
    def _copy(p, d):
        if Path(p).is_dir():                                          # a native iView image folder
            shutil.copytree(p, d / Path(p).name)
        else:
            shutil.copy2(p, d / "1.dcm")

    for a, p in s.files.items():
        d = tmp / f"WL_G{a:03d}"
        d.mkdir()
        _copy(p, d)
    if s.pf:
        d = tmp / "PF"
        d.mkdir()
        _copy(s.pf, d)
    return tmp


# ── Processed-image registry (SQLite) ─────────────────────────────────────────────────────────────────────────

def init_registry(conn) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS epid_processed (sop_uid TEXT PRIMARY KEY, machine TEXT, status TEXT, at TEXT)")


def processed_sops(conn, machine: str | None = None) -> dict:
    q = "SELECT sop_uid, status FROM epid_processed" + (" WHERE machine=?" if machine else "")
    return dict(conn.execute(q, (machine,) if machine else ()).fetchall())


def mark(conn, sops, machine: str, status: str = "processed") -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    for u in sops:
        if u:
            conn.execute("INSERT OR REPLACE INTO epid_processed(sop_uid, machine, status, at) VALUES (?,?,?,?)", (u, machine, status, now))
    conn.commit()


def classify(sessions: list[Session], done: dict) -> None:
    """Set each session's status from the registry: processed when all its images were saved, skipped when set aside."""
    for s in sessions:
        st = {done.get(u) for u in s.sops.values()}
        s.status = "processed" if st == {"processed"} else "skipped" if st and st <= {"processed", "skipped"} and "skipped" in st and "new" not in st and None not in st else "new"


# ── Machine logs (optional) ──────────────────────────────────────────────────────────────────────────────────

NAME_RE = re.compile(r"^(?P<patient>.+?)_(?P<machine>[^_]+)_(?P<beam>[^-_]+)-(?P<desc>.*)_(?P<ts>\d{14})\.rllf$", re.I)
T_GANTRY, T_COLL = "50010007", "50010008"
T_Y1, T_Y2 = "50010009", "5001000a"


@dataclass
class LogRecord:
    path: Path
    when: dt.datetime
    gantry: float | None
    coll: float | None
    aperture: dict | None


def _f(cur: dict, tag: str, kind: str):
    try:
        return float(cur.get(tag, {}).get(kind, ""))
    except ValueError:
        return None


def read_log(path: Path) -> LogRecord | None:
    """One delivery log: when, the gantry / collimator angles and, for a static beam, the aperture (MLC edges mean of the open leaf pairs, and the Y jaws)."""
    from rllf_parser import RLLFParser, _parse_frames
    path = Path(path)
    m = NAME_RE.match(path.name)
    try:
        frames = _parse_frames(path.read_bytes())
    except Exception:
        return None
    if not frames:
        return None
    cur: dict = {}
    for fr in frames:
        for tag, t, v in fr:
            cur.setdefault(tag, {})[t] = v
    when = None
    if m:
        try:
            when = dt.datetime.strptime(m["ts"], "%Y%m%d%H%M%S")
        except ValueError:
            pass
    when = when or dt.datetime.fromtimestamp(path.stat().st_mtime)
    rec = LogRecord(path, when, _f(cur, T_GANTRY, "R"), _f(cur, T_COLL, "R"), None)
    jaws = {}
    for k, tag in (("Y1", T_Y1), ("Y2", T_Y2)):
        r_, s_ = _f(cur, tag, "R"), _f(cur, tag, "S")
        if r_ is not None and s_ is not None:
            jaws[k] = (r_ * 10.0, s_ * 10.0)
    try:
        p = RLLFParser(str(path))
        sa, sb = p.bank_a_expect, p.bank_b_expect
        both = np.concatenate([sa, sb], axis=1)
        if float(np.nanmax(np.nanmax(both, axis=0) - np.nanmin(both, axis=0))) > 1.0:        # a moving beam has no single aperture
            return rec
        open_ = np.abs(np.nan_to_num(np.nanmedian(sb - sa, axis=0))) > 5.0
        n = sa.shape[0]
        if open_.any() and n >= 6 and "Y1" in jaws and "Y2" in jaws:
            fr_ = slice(n // 2, n - 2)
            med = lambda mat: float(np.nanmedian(np.nanmedian(mat[fr_][:, open_], axis=0)))
            rec.aperture = {"x1": med(p.bank_a_actual), "x2": med(p.bank_b_actual), "x1_set": med(sa), "x2_set": med(sb),
                            "y1": jaws["Y1"][0], "y2": jaws["Y2"][0], "y1_set": jaws["Y1"][1], "y2_set": jaws["Y2"][1]}
    except Exception:
        pass
    return rec


def aperture_shift(rec: LogRecord) -> dict | None:
    """The field-centre shift the log reports (actual minus set centre of the MLC and jaws), in image axes (mm: u right, v down), plus the sizes. None without an aperture.
    The collimator turns counter-clockwise seen from the source (IEC 61217), i.e. by -angle in (u, v): at 90 deg its +X axis points up the image."""
    a = rec.aperture
    if not a:
        return None
    cx, cy = (a["x1"] + a["x2"]) / 2, (a["y2"] - a["y1"]) / 2
    cxs, cys = (a["x1_set"] + a["x2_set"]) / 2, (a["y2_set"] - a["y1_set"]) / 2
    dx, dy, c = cx - cxs, cy - cys, np.radians(rec.coll or 0.0)
    return {"shift": (float(np.cos(c) * dx + np.sin(c) * dy), float(-np.sin(c) * dx + np.cos(c) * dy)), "width": a["x2"] - a["x1"], "height": a["y1"] + a["y2"],
            "width_set": a["x2_set"] - a["x1_set"], "height_set": a["y1_set"] + a["y2_set"], "file": rec.path.name}


class LogIndex:
    """The delivery logs of one folder (cached per file name + size), searchable by time."""

    def __init__(self, folder: str | Path, cache_dir: Path):
        self.folder = Path(folder)
        self.cache_file = Path(cache_dir) / "epid_log_cache.json"
        self.records: list[LogRecord] = []
        self.load()

    def load(self):
        try:
            cache = json.loads(self.cache_file.read_text())
        except (OSError, ValueError):
            cache = {}
        fresh = {}
        if self.folder.is_dir():
            for p in sorted(self.folder.rglob("*.rllf")):
                try:
                    st = p.stat()
                except OSError:
                    continue
                k = str(p)
                c = cache.get(k)
                if c and c.get("s") == st.st_size and c.get("m") == st.st_mtime:
                    rec = None if c.get("rec") is None else LogRecord(p, dt.datetime.fromisoformat(c["rec"]["when"]), c["rec"]["g"], c["rec"]["c"], c["rec"]["a"])
                else:
                    rec = read_log(p)
                fresh[k] = {"s": st.st_size, "m": st.st_mtime,
                            "rec": None if rec is None else {"when": rec.when.isoformat(), "g": rec.gantry, "c": rec.coll, "a": rec.aperture}}
                if rec:
                    self.records.append(rec)
        try:
            self.cache_file.write_text(json.dumps(fresh))
        except OSError:
            pass

    def match(self, images: dict) -> dict:
        """images: {angle: (datetime, gantry)}. Returns {angle: LogRecord} pairing each image with the delivery that made it (matched on gantry angle and time).
        The imager clock differs from the machine's by minutes and in either direction (+4.5 min one day, -7.7 min another): the offset is the one that lines up the
        most images with a log of the same gantry angle, then each image takes the nearest log to it."""
        def compat(r, g):
            return not (r.gantry is not None and g is not None and min(abs(r.gantry - g), 360 - abs(r.gantry - g)) > 10.0)

        def pick(offset_s, window_s):
            out = {}
            for a, (when, g) in images.items():
                best = None
                for r in self.records:
                    if not compat(r, g):
                        continue
                    d = abs((when - r.when).total_seconds() - offset_s)
                    if d <= window_s and (best is None or d < best[0]):
                        best = (d, r)
                if best:
                    out[a] = best[1]
            return out
        cands = sorted({round((when - r.when).total_seconds()) for a, (when, g) in images.items() for r in self.records
                        if compat(r, g) and abs((when - r.when).total_seconds()) <= 2700})
        if not cands:
            return {}
        def score(b):
            got = pick(b, 150.0)
            return (len(got), -float(np.mean([abs((images[a][0] - r.when).total_seconds() - b) for a, r in got.items()])) if got else 0.0)
        off = max(cands, key=score)
        first = pick(off, 300.0)
        if not first:
            return {}
        off = float(np.median([(images[a][0] - r.when).total_seconds() for a, r in first.items()]))
        return pick(off, 300.0) or first
