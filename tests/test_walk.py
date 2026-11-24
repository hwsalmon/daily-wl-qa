"""The 4-gantry-angle walk: static phantom offset with opposite signs at opposite angles, exact smallest sphere, pass/fail on the maximum displacement from the isocenter (TG-198 / MPPG: < 1 mm for SRS)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import wl_qa_tool as w


def images(X, Y, Z, walk=None):
    """Portal displacements of a phantom offset (X, Y, Z) plus an optional per-angle gantry walk {angle: (du, dv)}."""
    out = {}
    for a, (cx, cz) in w._GANTRY_U.items():
        du, dv = (walk or {}).get(a, (0.0, 0.0))
        out[a] = {"dx_mm": cx * X + cz * Z + du, "dy_mm": Y + dv}
    return out


def test_static_offset_is_removed_whatever_its_size():
    r = w.compute_wl_results(images(1.2, -0.7, 0.9))
    assert r["walk_circle_r"] == pytest.approx(0.0, abs=1e-4) and r["pass_fail"]
    assert (r["setup_x"], r["setup_y"], r["setup_z"]) == pytest.approx((1.2, -0.7, 0.9), abs=1e-3)


def test_reported_walk_is_the_max_displacement_and_pass_fail_uses_it():
    walk = {0: (0.3, 0.0), 180: (-0.3, 0.0), 90: (0.0, 0.0), 270: (0.0, 0.0)}         # a 0.3 mm walk in x at G0 / G180
    r = w.compute_wl_results(images(0.5, 0.2, -0.4, walk))
    assert r["max_2d_walk_mm"] == pytest.approx(r["walk_circle_r"]) and r["walk_diameter_mm"] == pytest.approx(2 * r["walk_circle_r"])     # the diameter is information only
    assert r["pass_fail"] == (r["max_2d_walk_mm"] <= w.TOLERANCE_MM)
    # a 1.2 mm G0-G180 split (diameter 1.2 mm) is 0.6 mm from the isocenter: a PASS; it takes a 2.2 mm split to fail
    ok = w.compute_wl_results(images(0.0, 0.0, 0.0, {0: (0.0, 0.6), 180: (0.0, -0.6), 90: (0, 0), 270: (0, 0)}))
    assert ok["max_2d_walk_mm"] == pytest.approx(0.6, abs=1e-3) and ok["pass_fail"]
    big = w.compute_wl_results(images(0.0, 0.0, 0.0, {0: (0.0, 1.1), 180: (0.0, -1.1), 90: (0, 0), 270: (0, 0)}))
    assert big["max_2d_walk_mm"] == pytest.approx(1.1, abs=1e-3) and not big["pass_fail"]


def test_matches_global_minimum_on_random_data():
    from scipy.optimize import differential_evolution
    rng = np.random.default_rng(3)
    for _ in range(15):
        img = {a: {"dx_mm": float(rng.normal(0, 0.8)), "dy_mm": float(rng.normal(0, 0.8))} for a in w.GANTRY_ANGLES}
        dx = {a: img[a]["dx_mm"] for a in img}; dy = {a: img[a]["dy_mm"] for a in img}
        r = w.compute_wl_results(img)["walk_circle_r"]
        g = differential_evolution(lambda p: w._wl_walk(p, dx, dy), [(-4, 4)] * 3, seed=1, tol=1e-12, maxiter=600, polish=False).fun
        assert r <= g + 1e-4


# ── void and field-edge estimators ───────────────────────────────────────────

def _synthetic_image(void_dx_mm=0.0, void_dy_mm=0.0, bowl=True, field_mm=40.0, sp=0.253):
    """1024 px image, low value = dose: bright background, a field_mm square with a sloping 'bowl' profile and a void shadow."""
    import numpy as np
    import wl_qa_tool as t
    n = 1024
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    cx = cy = 511.5
    X, Y = (xx - cx) * sp, (yy - cy) * sp
    def soft(d):                                    # 0 outside, 1 inside, ~3 mm penumbra
        return 1.0 / (1.0 + np.exp(-d / 0.6))
    inside = soft(field_mm / 2 - np.abs(X)) * soft(field_mm / 2 - np.abs(Y))
    plateau = 27000.0 - (2500.0 * np.exp(-(X ** 2 + Y ** 2) / (2 * 14.0 ** 2)) if bowl else 0.0)
    img = 64900.0 - inside * (64900.0 - plateau)
    r = np.hypot(X - void_dx_mm, Y - void_dy_mm)
    img -= 1500.0 * np.interp(r, np.arange(len(t._VOID_PROFILE)) * t._VOID_PROFILE_STEP_MM, t._VOID_PROFILE, right=0.0)
    return img.astype(np.float32), sp


def test_field_edges_read_the_true_field_size():
    import wl_qa_tool as t
    img, sp = _synthetic_image(0.8, -0.5)
    e = t.measure_field_edges(img, sp)
    assert abs(e["mlc_total_mm"] - 40.0) < 0.4 and abs(e["jaw_total_mm"] - 40.0) < 0.4


def test_void_fit_recovers_a_known_offset_without_pulling_to_beam_centre():
    import wl_qa_tool as t
    for dx, dy in ((0.0, 0.0), (1.5, 0.0), (-1.2, 1.0), (0.0, -2.0)):
        img, sp = _synthetic_image(dx, dy)
        e = t.measure_field_edges(img, sp)
        f = t.fit_void(img, sp, e["edge_center_px"])
        assert f["ok"] and f["method"] == "fit"
        got_dx = (f["col"] - 511.5) * sp
        got_dy = (f["row"] - 511.5) * sp
        assert abs(got_dx - dx) < 0.15 and abs(got_dy - dy) < 0.15


def test_void_fit_reports_a_missing_void():
    import numpy as np
    import wl_qa_tool as t
    img, sp = _synthetic_image(0.0, 0.0)
    img = img + 0.0
    img[420:600, 420:600] = img[420:600, 420:600].mean()      # blank out the void region (no dip at all)
    f = t.fit_void(img, sp, (511.5, 511.5))
    assert not f["ok"]


# ── loading and saving ───────────────────────────────────────────────────────

def _dcm(path, gantry, when, value):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian; meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.1"; meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.GantryAngle = gantry; ds.ContentDate, ds.ContentTime = when
    ds.Rows = ds.Columns = 4; ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = (np.full((4, 4), value, np.uint16)).tobytes(); ds.is_little_endian = True; ds.is_implicit_VR = False
    ds.save_as(str(path), write_like_original=False)


def test_a_repeated_angle_takes_the_latest_image_whatever_the_file_order(tmp_path):
    for k, (g, t, v) in enumerate(((0, "120000", 1), (90, "120100", 2), (180, "120200", 3), (270, "120300", 4), (0, "120500", 99))):
        _dcm(tmp_path / (f"z{k}.dcm" if k == 4 else f"a{k}.dcm"), g, ("20261123", t), v)         # the NEWER G0 image has the file name that sorts first or last: neither may decide
    _dcm(tmp_path / "0_old.dcm", 0, ("20261123", "110000"), 55)                                 # an older G0 image that sorts before everything
    imgs = w.load_dicom_images(str(tmp_path))
    assert imgs[0].pixel_array[0, 0] == 99 and set(imgs) == {0, 90, 180, 270}


def test_saving_the_same_session_twice_keeps_one_record(tmp_path, monkeypatch):
    import datetime, sqlite3
    monkeypatch.setattr(w, "DB_PATH", tmp_path / "h.db"); w._init_db()
    r = w.compute_wl_results(images(0.2, 0.1, -0.1, {0: (0.0, 0.3), 180: (0.0, -0.3), 90: (0, 0), 270: (0, 0)}))
    ir = {a: {"field_edges": {}, "sop_uid": f"u{a}"} for a in w.GANTRY_ANGLES}
    when = datetime.datetime(2026, 11, 23, 16, 19)
    for _ in range(2):
        w._save_to_db(r, ir, "Elekta VersaHD 154613", "HWS", when=when)
    n, = sqlite3.connect(tmp_path / "h.db").execute("SELECT COUNT(*) FROM wl_records").fetchone()
    assert n == 1


def test_a_failed_load_does_not_leave_the_previous_session_to_be_reported(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(w, "DB_PATH", tmp_path / "h.db"); monkeypatch.setattr(w, "CONFIG_PATH", tmp_path / "c.json")
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    win = w.WLApp()
    win._wl_results = {"x": 1}; win._image_results = {"x": 1}; win._report_btn.setEnabled(True)
    (tmp_path / "empty").mkdir()
    win._process_directory(str(tmp_path / "empty"))
    assert win._wl_results is None and not win._report_btn.isEnabled()


def test_a_folder_of_several_sessions_analyses_the_newest_complete_one_and_never_mixes_them(tmp_path):
    def pid(path, g, when, v, patient):
        _dcm(path, g, when, v)
        import pydicom
        ds = pydicom.dcmread(str(path), force=True); ds.PatientID = patient; ds.save_as(str(path), write_like_original=False)
    for k, (g, t) in enumerate(((180, "071347"), (90, "071428"), (270, "071524"), (0, "071606"))):
        pid(tmp_path / f"v2_{k}.dcm", g, ("20260701", t), 10 + k, "QA_Daily_V2_26")
    for k, (g, t) in enumerate(((180, "104847"), (90, "104941"), (270, "105041"), (0, "105126"))):
        pid(tmp_path / f"mv_{k}.dcm", g, ("20260701", t), 20 + k, "QA_Daily_MV_26")
    for k, (g, t) in enumerate(((180, "110000"), (90, "110100"))):                                # the newest session is incomplete: it is not used
        pid(tmp_path / f"v1_{k}.dcm", g, ("20260701", t), 30 + k, "QA_Daily_V1_26")
    imgs = w.load_dicom_images(str(tmp_path))
    assert {a: int(ds.pixel_array[0, 0]) for a, ds in imgs.items()} == {180: 20, 90: 21, 270: 22, 0: 23} and "newest of 3 sessions" in w.load_dicom_images.note
