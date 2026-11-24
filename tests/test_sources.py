"""EPID-folder scan: Daily WL sessions from a mixed folder, repeats, the processed registry; log pairing by gantry angle and time."""
import pytest
import datetime as dt
import os
import sqlite3
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

import wl_sources as s


def write(path, label, gantry, when, patient="QA_Daily_V2_26", series=None):
    uid = generate_uid()
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.1"; meta.MediaStorageSOPInstanceUID = uid
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPInstanceUID = uid; ds.Modality = "RTIMAGE"; ds.RTImageLabel = label; ds.SeriesDescription = series or f"{label} + x"
    ds.PatientID = patient; ds.GantryAngle = gantry
    ds.ContentDate, ds.ContentTime = when.strftime("%Y%m%d"), when.strftime("%H%M%S")
    ds.Rows = ds.Columns = 64; ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0; ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"; ds.PixelData = np.zeros((64, 64), np.uint16).tobytes() * 3          # > 10 kB like a real image
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)
    return str(ds.SOPInstanceUID)


def make(tmp):
    t0 = dt.datetime(2026, 7, 1, 7, 0, 0)
    sops = {}
    for i, g in enumerate((180, 90, 270, 0)):
        sops[g] = write(tmp / "a" / f"{i}" / "1.dcm", f"WL{i + 1}", g + 0.2, t0 + dt.timedelta(seconds=90 * i))
    write(tmp / "junk" / "1.dcm", "6MV", 0.0, t0 + dt.timedelta(minutes=10))                        # another daily-QA image: ignored
    write(tmp / "pf" / "1.dcm", "PF1", 0.0, t0 + dt.timedelta(minutes=20), series="PF test")        # the day's picket fence image
    t1 = dt.datetime(2026, 7, 2, 7, 5, 0)
    for i, g in enumerate((180, 90, 270)):                                                          # next day: only three angles
        write(tmp / "b" / f"{i}" / "1.dcm", f"WL{i + 1}", g - 0.1, t1 + dt.timedelta(seconds=90 * i))
    return sops


def test_sessions_pf_and_incomplete(tmp_path):
    sops = make(tmp_path)
    ses = s.scan_epid_folder(tmp_path, tmp_path / "cache")
    assert len(ses) == 2 and ses[0].when.day == 2                                    # newest first
    new, old = ses[0], ses[1]
    assert not new.complete and new.missing == [0] and old.complete and old.pf is not None
    assert {a: old.sops[a] for a in s.CARDINALS} == sops
    again = s.scan_epid_folder(tmp_path, tmp_path / "cache")                         # served from the header cache
    assert [x.key for x in again] == [x.key for x in ses]


def test_repeat_replaces_and_registry(tmp_path):
    make(tmp_path)
    write(tmp_path / "rep" / "1.dcm", "WL4", 0.1, dt.datetime(2026, 7, 1, 7, 12, 0))                # a repeated G0 delivery 12 min later: the latest is used
    ses = s.scan_epid_folder(tmp_path, tmp_path / "cache")
    day1 = next(x for x in ses if x.when.day == 1)
    assert day1.files[0].parent.name == "rep"
    conn = sqlite3.connect(":memory:"); s.init_registry(conn)
    s.classify(ses, s.processed_sops(conn)); assert all(x.status == "new" for x in ses)
    s.mark(conn, day1.sops.values(), "M", "processed"); s.classify(ses, s.processed_sops(conn))
    assert day1.status == "processed" and next(x for x in ses if x.when.day == 2).status == "new"
    st = s.stage_session(day1)
    assert len(list(st.rglob("1.dcm"))) == 5                                             # four WL images + the PF image


def test_log_matching_by_gantry_and_time(tmp_path):
    base = dt.datetime(2026, 7, 1, 7, 0, 0)
    idx = s.LogIndex.__new__(s.LogIndex)
    idx.records = [s.LogRecord(tmp_path / f"l{g}.rllf", base + dt.timedelta(seconds=100 * i) - dt.timedelta(seconds=270), float(g), 0.0, None) for i, g in enumerate((180, 90, 270, 0))]
    imgs = {g: (base + dt.timedelta(seconds=100 * i), g + 0.2) for i, g in enumerate((180, 90, 270, 0))}      # imager clock 270 s ahead
    got = idx.match(imgs)
    assert {a: r.gantry for a, r in got.items()} == {180: 180.0, 90: 90.0, 270: 270.0, 0: 0.0}
    assert idx.match({0: (base + dt.timedelta(days=3), 0.0)}) == {}                  # nothing within the window: no match, no error


# ── native iView export (img_<uid> folders: lossless JPEG + _Frame.xml) ──────────────────────────────────────────

def _field(size_mm, dip=True, sp=0.25, n=1024, long_mm=None):
    import numpy as np
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    X, Y = (xx - n / 2) * sp, (yy - n / 2) * sp
    hx, hy = size_mm / 2, (long_mm or size_mm) / 2
    inside = ((np.abs(X) <= hx) & (np.abs(Y) <= hy)).astype(float)
    img = 65000.0 - inside * 38000.0
    if dip:
        img -= inside * 2500.0 * np.exp(-(X ** 2 + Y ** 2) / (2 * 3.0 ** 2))
    return img.astype(np.float32)


def test_native_classifier_tells_wl_picket_and_others_apart():
    import iview_native as nv
    assert nv.measure(_field(40))["kind"] == "wl"                       # Daily WL: 40 x 40 mm with the phantom's void
    assert nv.measure(_field(14, long_mm=200))["kind"] == "pf"          # Daily picket fence: one 1.4 cm x 20 cm picket
    assert nv.measure(_field(30))["kind"] == "other"                    # the monthly BB field is 30 x 30 mm
    assert nv.measure(_field(150, dip=False))["kind"] == "other"        # field-size image
    assert nv.measure(_field(13, long_mm=125))["kind"] == "other"       # collimator slit (Starshot)


def test_logs_are_matched_when_the_imager_clock_is_behind_the_machine():
    import datetime as dt
    from pathlib import Path
    import wl_sources as ws
    idx = ws.LogIndex.__new__(ws.LogIndex)
    day = lambda h, m, s: dt.datetime(2026, 11, 22, h, m, s)
    idx.records = [ws.LogRecord(Path(f"WL{k}.rllf"), day(*t), g, 0.0, None) for k, (t, g) in enumerate(
        [((16, 25, 43), 180.0), ((16, 26, 37), 90.0), ((16, 27, 45), -90.0), ((16, 28, 40), -0.2), ((16, 29, 15), -0.2)], 1)]
    images = {180: (day(16, 18, 2), 180.0), 90: (day(16, 18, 55), 90.0), 270: (day(16, 20, 3), 270.0), 0: (day(16, 20, 56), 359.8)}
    got = idx.match(images)
    assert {a: r.path.name for a, r in got.items()} == {180: "WL1.rllf", 90: "WL2.rllf", 270: "WL3.rllf", 0: "WL4.rllf"}      # the PF log (same angle as WL4) is not taken
    idx.records = [ws.LogRecord(r.path, r.when - dt.timedelta(seconds=800), r.gantry, r.coll, None) for r in idx.records]       # imager 5 min ahead instead
    got = idx.match(images)
    assert {a: r.path.name for a, r in got.items()} == {180: "WL1.rllf", 90: "WL2.rllf", 270: "WL3.rllf", 0: "WL4.rllf"}


def test_native_session_end_to_end(tmp_path):
    import pytest
    from pathlib import Path
    import wl_qa_tool as t
    import wl_sources as ws
    d = Path(__file__).resolve().parents[1] / "Test Data" / "VersaHD 1 w log files"
    if not d.is_dir():
        pytest.skip("native test session not present")
    imgs = t.load_dicom_images(str(d))
    assert sorted(imgs) == [0, 90, 180, 270] and imgs[0].PatientID == "QA_Daily_V1_26"      # the patient id comes from the logs' file names
    res = {a: t.analyze_image(ds) for a, ds in imgs.items()}
    assert all(r["void_fit"]["ok"] for r in res.values()) and abs(res[0]["field_edges"]["mlc_total_mm"] - 40.0) < 0.8
    idx = ws.LogIndex(str(d), tmp_path)
    got = idx.match({a: (r["when"], r["gantry"]) for a, r in res.items()})
    assert len(got) == 4
    sess = ws.scan_epid_folder(str(d), tmp_path)
    assert len(sess) == 1 and sess[0].complete and sess[0].pf is not None
    staged = ws.stage_session(sess[0])
    assert sorted(t.load_dicom_images(str(staged))) == [0, 90, 180, 270]


def test_aperture_shift_turns_with_the_collimator_counter_clockwise():
    from pathlib import Path
    import datetime as dt
    import wl_sources as ws
    ap = {"x1": -15.0 + 0.2, "x2": 15.0 + 0.2, "x1_set": -15.0, "x2_set": 15.0, "y1": 15.0 + 0.1, "y2": 15.0 - 0.1, "y1_set": 15.0, "y2_set": 15.0}      # MLC centre +0.2, jaw centre -0.1
    mk = lambda c: ws.LogRecord(Path("x.rllf"), dt.datetime(2026, 11, 22), 0.0, c, ap)
    assert ws.aperture_shift(mk(0.0))["shift"] == pytest.approx((0.2, -0.1))
    assert ws.aperture_shift(mk(90.0))["shift"] == pytest.approx((-0.1, -0.2))                  # +X points up the image at 90 deg
    assert ws.aperture_shift(mk(180.0))["shift"] == pytest.approx((-0.2, 0.1))
    assert ws.aperture_shift(mk(270.0))["shift"] == pytest.approx((0.1, 0.2))


def test_a_saved_record_carries_the_acquisition_time_not_the_saving_time(tmp_path, monkeypatch):
    import datetime as dt
    import sqlite3
    import wl_qa_tool as t
    monkeypatch.setattr(t, "DB_PATH", str(tmp_path / "h.db"))
    t._init_db()
    pa = {a: {"raw_dx": 0.1, "raw_dy": 0.2} for a in (0, 90, 180, 270)}
    wl = {"per_angle": pa, "walk_circle_r": 0.6, "pass_fail": False, "baseline_dx": 0.0, "baseline_dy": 0.0, "baseline_dz": 0.0, "logs_used": 0}
    ir = {a: {"field_edges": {"mlc_total_mm": 40.0, "jaw_total_mm": 40.1}, "sop_uid": f"u{a}"} for a in (0, 90, 180, 270)}
    t._save_to_db(wl, ir, "Elekta VersaHD 153991", "X", when=dt.datetime(2026, 11, 22, 16, 18))
    t._save_to_db(wl, ir, "Elekta VersaHD 153991", "X")                                     # no time known: the saving time is the fallback
    rows = sqlite3.connect(tmp_path / "h.db").execute("select date, time from wl_records order by id").fetchall()
    assert rows[0] == ("2026-11-22", "16:18") and rows[1][0] == dt.date.today().isoformat()
