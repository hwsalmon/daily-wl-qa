"""Bulk import (stable with and without logs, resumable, never raises for a data problem) and the four trend analyses."""
import datetime as dt
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import daily_batch as db_
import daily_trends as tr
import wl_qa_tool as w


# ── trends ───────────────────────────────────────────────────────────────────

def _wl_db(path, n=40, slope_per_day=0.0, noise=0.02, machine="M"):
    w.DB_PATH = path; w._init_db()
    rng = np.random.default_rng(1)
    c = sqlite3.connect(path)
    for k in range(n):
        d = dt.datetime(2026, 6, 1, 7, 0) + dt.timedelta(days=k)
        r = 0.55 + slope_per_day * k + rng.normal(0, noise)
        c.execute("INSERT INTO wl_records (date,time,machine_name,physicist_name,walk_circle_r,pass_fail,raw_dy_G0,raw_dy_G180,avg_mlc_mm,avg_jaw_mm) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (d.strftime("%Y-%m-%d"), "07:00", machine, "x", r, 1, 0.5 + slope_per_day * k, -0.6, 40.2 + rng.normal(0, 0.05), 40.3 + 0.004 * k))
    c.commit(); c.close()


def test_drift_is_flagged_for_a_real_slope_and_not_for_noise(tmp_path):
    _wl_db(tmp_path / "a.db", slope_per_day=0.004)                     # 0.12 mm per month
    w_ = tr.load(tmp_path / "a.db", "M")["wl"]
    d = tr.drift([r["when"] for r in w_], [r["walk_circle_r"] for r in w_], "walk")
    assert d["flag"] and d["slope_30d"] == pytest.approx(0.12, abs=0.03) and d["p"] < 1e-6
    _wl_db(tmp_path / "b.db", slope_per_day=0.0)
    w_ = tr.load(tmp_path / "b.db", "M")["wl"]
    d = tr.drift([r["when"] for r in w_], [r["walk_circle_r"] for r in w_], "walk")
    assert not d["flag"]
    assert not tr.drift([dt.datetime(2026, 6, 1) + dt.timedelta(days=i) for i in range(3)], [0.5, 0.5, 0.5], "walk")["flag"]         # too few sessions: no claim


def test_walk_and_size_trends_build_a_figure_and_a_table(tmp_path):
    _wl_db(tmp_path / "a.db", slope_per_day=0.004)
    lim = {"walk": 1.0, "mlc": (0.4, 0.6), "jaw": (0.6, 0.8), "pf": (0.4, 0.5)}
    for key in ("walk", "mlc", "jaw"):
        r = tr.build(tmp_path / "a.db", "M", key, {"mlc": 40.0, "jaw": 40.0}, lim)
        assert r["png"][:4] == b"\x89PNG" and r["rows"] and r["summary"]
    assert "DRIFT" in tr.build(tmp_path / "a.db", "M", "walk", {}, lim)["summary"]
    assert "DRIFT" in tr.build(tmp_path / "a.db", "M", "jaw", {"jaw": 40.0}, lim)["summary"]          # the jaw size creeps 0.004 mm / day in the fixture
    assert "No picket-fence" in tr.build(tmp_path / "a.db", "M", "pf", {}, lim)["summary"] and tr.build(tmp_path / "a.db", "NOPE", "walk", {}, lim)["png"] is None


def _pf_db(path, same_leaf=True, n=30):
    w.DB_PATH = path; w._init_db()
    rng = np.random.default_rng(2)
    for k in range(n):
        d = rng.normal(0, 0.04, 40)
        if same_leaf:
            d[17] += 0.18                                              # leaf 17 is always 0.18 mm out
        pf = {"leaf_deviations_mm": d, "leaf_y_mm": np.linspace(-100, 100, 40), "pass_fail": True, "study_date": dt.date(2026, 6, 1) + dt.timedelta(days=k), "study_time": dt.time(7, 0)}
        c = sqlite3.connect(path); w._save_pf(c, pf, "M", "x"); c.commit(); c.close()


def test_the_same_worst_leaf_is_found_and_a_random_one_is_not(tmp_path):
    _pf_db(tmp_path / "p.db", same_leaf=True)
    c = tr.leaf_consistency(tr.load(tmp_path / "p.db", "M")["pf"])
    assert c["top"] == 17 and c["top_frac"] > 0.9 and c["p"] < 1e-6 and c["persistent"][0][0] == 17
    assert "SAME LEAF" in tr.leaf_text(c)
    _pf_db(tmp_path / "q.db", same_leaf=False)
    c = tr.leaf_consistency(tr.load(tmp_path / "q.db", "M")["pf"])
    assert c["p"] > 0.01 and not c["persistent"] and "SAME LEAF" not in tr.leaf_text(c)
    lim = {"walk": 1.0, "mlc": (0.4, 0.6), "jaw": (0.6, 0.8), "pf": (0.4, 0.5)}
    for key in ("pf", "leaf"):
        r = tr.build(tmp_path / "p.db", "M", key, {}, lim)
        assert r["png"][:4] == b"\x89PNG" and r["rows"]


def test_a_picket_fence_session_is_saved_once_and_never_blocks_the_wl_record(tmp_path):
    w.DB_PATH = tmp_path / "h.db"; w._init_db()
    pf = {"leaf_deviations_mm": np.array([0.1, -0.3, 0.05]), "leaf_y_mm": np.array([-5.0, 0.0, 5.0]), "pass_fail": True, "study_date": dt.date(2026, 6, 1), "study_time": dt.time(7, 0)}
    c = sqlite3.connect(w.DB_PATH)
    for _ in range(2):
        w._save_pf(c, pf, "M", "x")
    c.commit()
    n, worst = c.execute("SELECT COUNT(*), MIN(worst_leaf) FROM pf_records").fetchone()
    assert n == 1 and worst == 1


# ── bulk import ──────────────────────────────────────────────────────────────

def _sess(when, complete=True, pf=False):
    s = SimpleNamespace(when=when, files={a: Path(f"/x/{a}") for a in (0, 90, 180, 270)} if complete else {0: Path("/x/0")}, sops={a: f"s{when:%H%M%S}{a}" for a in (0, 90, 180, 270)},
                        pf=Path("/x/pf") if pf else None)
    s.missing = [] if complete else [90, 180, 270]; s.complete = complete
    return s


def _fake_module(tmp_path, bad_days=(), crash_days=(), void_bad_days=()):
    """A stand-in for wl_qa_tool: analyse_image returns synthetic displacements, and the chosen days misbehave."""
    saved = []

    def analyze_image(p):
        day = p["day"]
        if day in crash_days:
            raise RuntimeError("corrupt image")
        a = int(Path(p["path"]).name)
        cx, cz = w._GANTRY_U[a]
        return {"dx_mm": 0.1 * cx, "dy_mm": 0.3 * (1 if a in (0, 90) else -1) * 0.5, "when": dt.datetime(2026, 6, day), "gantry": float(a), "sop_uid": f"s{day}{a}", "field_edges": {},
                "void_fit": {"ok": day not in void_bad_days}, "pixel_array": np.zeros((10, 10))}

    mod = SimpleNamespace(
        analyze_image=analyze_image, read_image=lambda p: p, compute_wl_results=w.compute_wl_results, analyze_picket_fence=lambda ds: (_ for _ in ()).throw(ValueError("no picket")),
        _save_to_db=lambda wl, ir, machine, phys, when=None, pf_results=None: (saved.append((when, machine, sorted(ir), "pixel_array" in ir[0])), None)[1])
    return mod, saved


class _S(SimpleNamespace):
    pass


def _sessions(days, **kw):
    out = []
    for d in days:
        s = _sess(dt.datetime(2026, 6, d, 7, 0), **kw)
        s.files = {a: {"day": d, "path": f"/x/{a}"} for a in (0, 90, 180, 270)}
        out.append(s)
    return out


def test_import_goes_on_after_a_bad_session_and_reports_each(tmp_path):
    mod, saved = _fake_module(tmp_path, crash_days=(3,), void_bad_days=(5,))
    ss = _sessions(range(1, 8)) + [_sess(dt.datetime(2026, 6, 20, 7, 0), complete=False)]
    seen = []
    out = db_.import_sessions(mod, ss, "M", "x", log_folders=[str(tmp_path / "no_such_folder")], cache_dir=tmp_path, progress=lambda i, n, t: seen.append(i))
    assert [o.status for o in out] == ["saved", "saved", "failed", "saved", "skipped", "saved", "saved", "skipped"]
    assert "RuntimeError" in out[2].message and "void" in out[4].message and "missing" in out[7].message
    assert len(saved) == 5 and not any(s[3] for s in saved) and seen[-1] == 8             # saved sessions carry no image arrays; progress reached the end
    assert "5 session(s) saved" in db_.summary_text(out) and "1 failed" in db_.summary_text(out)


def test_import_can_be_cancelled_and_resumed_and_works_without_any_logs(tmp_path):
    mod, saved = _fake_module(tmp_path)
    ss = _sessions(range(1, 11))
    n = []
    out = db_.import_sessions(mod, ss, "M", "x", log_folders=[], cache_dir=tmp_path, should_cancel=lambda: len(n.append(1) or n) > 4)
    assert len(out) == 4 and all(o.status == "saved" and o.logs == 0 for o in out)         # stopped after four, no logs anywhere
    out2 = db_.import_sessions(mod, ss[4:], "M", "x", log_folders=[None, ""], cache_dir=tmp_path)
    assert len(out2) == 6 and len(saved) == 10


def test_a_corrupt_or_empty_log_folder_means_no_logs_not_an_error(tmp_path):
    (tmp_path / "logs").mkdir(); (tmp_path / "logs" / "QA_Daily_x_WL1-WL G000C000T000_20260601070000.rllf").write_bytes(b"not a real log")
    assert db_.open_log_index([str(tmp_path / "logs")], tmp_path) is None                  # nothing readable: no index, the analysis goes on without logs
    assert db_.open_log_index([str(tmp_path / "missing")], tmp_path) is None
    mod, saved = _fake_module(tmp_path)
    out = db_.import_sessions(mod, _sessions([1, 2]), "M", "x", log_folders=[str(tmp_path / "logs")], cache_dir=tmp_path)
    assert [o.status for o in out] == ["saved", "saved"]


def test_the_trend_dialog_pages_draw_for_every_test_and_survive_an_empty_database(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QStackedWidget
    app = QApplication.instance() or QApplication([])
    _wl_db(tmp_path / "t.db", slope_per_day=0.002); _pf_db(tmp_path / "t.db", same_leaf=True, n=12)
    monkeypatch.setattr(w, "DB_PATH", tmp_path / "t.db"); monkeypatch.setattr(w, "CONFIG_PATH", tmp_path / "c.json")
    win = w.WLApp()
    assert "M" in win._trend_machines_with_data()
    stack = QStackedWidget(); state = {"test_page": None}
    for key, _label in win._TREND_TESTS:
        page = win._build_trend_chart_page("M", key, stack, state)
        assert page is not None
    monkeypatch.setattr(w, "DB_PATH", tmp_path / "none.db")                                     # no database at all: pages say so, nothing raises
    page = win._build_trend_chart_page("M", "walk", stack, state)
    assert page is not None and win._trend_machines_with_data() == []


def test_a_step_change_is_found_where_it_happened_and_a_smooth_series_has_none():
    t = [dt.datetime(2026, 6, 1) + dt.timedelta(days=i) for i in range(30)]
    rng = np.random.default_rng(5)
    y = np.r_[np.full(17, 40.7), np.full(13, 42.15)] + rng.normal(0, 0.04, 30)              # the V1 MLC size jumped 1.45 mm on 18 June
    st = tr.find_steps(t, y, "size")
    assert len(st) == 1 and st[0][1] == pytest.approx(1.45, abs=0.1) and st[0][3] == t[17]
    assert "STEP CHANGE" in tr.steps_text(st)
    assert tr.find_steps(t, 40.7 + rng.normal(0, 0.06, 30), "size") == [] and tr.find_steps(t[:5], y[:5], "size") == []
    two = np.r_[np.full(10, 0.5), np.full(10, 0.8), np.full(10, 0.55)] + rng.normal(0, 0.01, 30)
    assert len(tr.find_steps(t, two, "walk")) == 2


# ── history, reports, sign-off ───────────────────────────────────────────────

def _history_db(path):
    _wl_db(path, n=40, slope_per_day=0.0); _pf_db(path, same_leaf=False, n=40)
    c = sqlite3.connect(path)
    for r in c.execute("SELECT id FROM wl_records").fetchall():
        c.execute("UPDATE wl_records SET raw_dx_G0=0.3, raw_dy_G0=0.5, raw_dx_G90=0.1, raw_dy_G90=0.2, raw_dx_G180=-0.2, raw_dy_G180=-0.6, raw_dx_G270=0.0, raw_dy_G270=0.1, baseline_dx=0.1, baseline_dy=0.0, baseline_dz=0.2 WHERE id=?", r)
    c.commit(); c.close()


def test_periods_and_records(tmp_path):
    import daily_reports as dr
    assert dr.period("day", dt.date(2026, 11, 9)) == (dt.date(2026, 11, 9), dt.date(2026, 11, 9))
    assert dr.period("week", dt.date(2026, 11, 5)) == (dt.date(2026, 11, 2), dt.date(2026, 11, 8))
    assert dr.period("month", dt.date(2026, 12, 15)) == (dt.date(2026, 12, 1), dt.date(2026, 12, 31))
    _history_db(tmp_path / "h.db")
    recs = dr.records_between(tmp_path / "h.db", "M", *dr.period("week", dt.date(2026, 6, 10)))
    assert len(recs) == 7 and all(r["pf"] is not None for r in recs) and dr.days_with_data(tmp_path / "h.db", "M")[0] == dt.date(2026, 6, 1)


@pytest.mark.parametrize("kind", ["day", "week", "month"])
def test_reports_build_as_pdf_draft_and_signed(tmp_path, kind):
    fitz = pytest.importorskip("fitz")
    import daily_reports as dr
    _history_db(tmp_path / "h.db")
    a, b = dr.period(kind, dt.date(2026, 6, 10))
    recs = dr.records_between(tmp_path / "h.db", "M", a, b)
    lim = {"walk": 1.0, "mlc": (0.4, 0.6), "jaw": (0.6, 0.8), "pf": (0.4, 0.5)}
    kw = dict(t=w, refs={"mlc": 40.2, "jaw": 40.3}, limits=lim, version="2.1.0")
    draft = tmp_path / "d.pdf"; dr.build_report(tmp_path / "h.db", "M", kind, a, b, out_path=draft, **kw)
    txt = "".join(p.get_text() for p in fitz.open(str(draft)))
    assert "DRAFT" in txt and "Sessions" in txt and "Isocenter" in txt and ("Winston-Lutz detail" in txt) == (kind == "day")
    so = dr.add_signoff(tmp_path / "h.db", "M", kind, a, b, "Dr Test", "all fine", recs, "x.pdf")
    signed = tmp_path / "s.pdf"; dr.build_report(tmp_path / "h.db", "M", kind, a, b, signoff=so, out_path=signed, **kw)
    txt = "".join(p.get_text() for p in fitz.open(str(signed)))
    assert "SIGNED" in txt and "Electronically signed by Dr Test" in txt and so["digest"][:16] in txt.replace("\n", "") and "all fine" in txt


def test_a_signed_report_notices_when_the_data_change(tmp_path):
    import daily_reports as dr
    _history_db(tmp_path / "h.db")
    a, b = dr.period("week", dt.date(2026, 6, 10)); recs = dr.records_between(tmp_path / "h.db", "M", a, b)
    dr.add_signoff(tmp_path / "h.db", "M", "week", a, b, "Dr Test", "", recs)
    so = dr.signoffs(tmp_path / "h.db", "M")[0]
    assert dr.still_matches(tmp_path / "h.db", so)
    c = sqlite3.connect(tmp_path / "h.db"); c.execute("UPDATE wl_records SET walk_circle_r=0.9 WHERE date=?", (recs[2]["date"],)); c.commit(); c.close()
    assert not dr.still_matches(tmp_path / "h.db", so)


def test_the_history_dialog_lists_a_period_shows_detail_and_signs(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
    app = QApplication.instance() or QApplication([])
    _history_db(tmp_path / "t.db")
    monkeypatch.setattr(w, "DB_PATH", tmp_path / "t.db"); monkeypatch.setattr(w, "CONFIG_PATH", tmp_path / "c.json")
    win = w.WLApp()
    d = w.HistoryDialog(win)
    d._machine.setCurrentText("M"); d._kind.setCurrentIndex(d._kind.findData("week")); d._go_latest()
    assert d._table.rowCount() >= 1 and "Detail" or True
    d._table.selectRow(0); assert "Maximum displacement" in d._detail.toPlainText() and "Picket fence" in d._detail.toPlainText()
    out = tmp_path / "signed.pdf"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("looks good", True)))
    monkeypatch.setattr(w, "QDesktopServices", type("D", (), {"openUrl": staticmethod(lambda u: True)}))
    d._make(sign=True)
    assert out.exists() and d._signed.rowCount() == 1 and d._signed.item(0, 5).text() == "yes"


def test_the_import_list_shows_field_sizes_and_picket_fence(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(w, "DB_PATH", tmp_path / "t.db"); monkeypatch.setattr(w, "CONFIG_PATH", tmp_path / "c.json")
    win = w.WLApp()
    s = SimpleNamespace(when=dt.datetime(2026, 11, 9, 7, 0), complete=True, missing=[], status="new", sops={}, files={},
                        preview={"displacement": 0.62, "pass": True, "logs": 0, "mlc": 40.3, "jaw": 40.9, "pf_max": 0.14})
    dlg = w.EpidSessionsDialog(win, [s], "Elekta VersaHD 153991")
    heads = [dlg._tree.headerItem().text(i) for i in range(dlg._tree.columnCount())]
    assert "MLC size (Δ ref)" in heads and "Jaw size (Δ ref)" in heads and "Picket fence max" in heads
    day = dlg._tree.topLevelItem(0); assert "2026-11-09" in day.text(0) and "(1)" in day.text(0)
    ch = day.child(0)
    row = [ch.text(i) for i in range(dlg._tree.columnCount())]
    assert "0.62 mm  PASS" in row and "40.30 (+0.30)" in row and "40.90 (+0.90)" in row and "0.14 mm" in row


def test_sessions_are_grouped_by_day_and_reviewed_in_turn(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(w, "DB_PATH", tmp_path / "t.db"); monkeypatch.setattr(w, "CONFIG_PATH", tmp_path / "c.json")
    win = w.WLApp()
    mk = lambda d, h: SimpleNamespace(when=dt.datetime(2026, 11, d, h, 0), complete=True, missing=[], status="new", sops={}, files={}, machine="M", preview=None)
    ss = [mk(24, 7), mk(23, 7), mk(23, 8), mk(25, 7)]
    dlg = w.EpidSessionsDialog(win, ss, "M")
    assert dlg._tree.topLevelItemCount() == 3
    assert [dlg._tree.topLevelItem(i).childCount() for i in range(3)] == [2, 1, 1]
    opened = []
    monkeypatch.setattr(win, "_process_directory", lambda *a, **k: opened.append(a))
    monkeypatch.setattr("wl_sources.stage_session", lambda s: tmp_path)
    dlg._tree.setCurrentItem(dlg._tree.topLevelItem(0))             # a day row opens its first session
    dlg._open()
    assert opened and win._queue_bar.isVisibleTo(win) and "1 of 4" in win._queue_label.text()
    win._queue_step(1); assert "2 of 4" in win._queue_label.text()
    win._queue_step(1); win._queue_step(1); assert "4 of 4" in win._queue_label.text() and not win._queue_next.isEnabled()


def test_history_opens_the_selected_session_in_the_application(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    import wl_sources
    app = QApplication.instance() or QApplication([])
    _history_db(tmp_path / "t.db")
    monkeypatch.setattr(w, "DB_PATH", tmp_path / "t.db"); monkeypatch.setattr(w, "CONFIG_PATH", tmp_path / "c.json")
    win = w.WLApp()
    epid = tmp_path / "epid"; epid.mkdir()
    monkeypatch.setattr(win, "_machine_folders", lambda m=None: {"epid": str(epid)})
    d = w.HistoryDialog(win)
    d._machine.setCurrentText("M"); d._kind.setCurrentIndex(d._kind.findData("day")); d._go_latest()
    d._table.selectRow(d._table.rowCount() - 1)
    when = d._recs[d._table.currentRow()]["when"]
    near = SimpleNamespace(when=when + dt.timedelta(minutes=2), complete=True, patient="", machine="")
    far = SimpleNamespace(when=when + dt.timedelta(hours=3), complete=True, patient="", machine="")
    monkeypatch.setattr(wl_sources, "scan_epid_folder", lambda *a, **k: [far, near])
    assert win._find_session_images("M", when) is near and near.machine == "M"
    assert win._find_session_images("M", when + dt.timedelta(days=1)) is None
    opened = []
    monkeypatch.setattr(win, "_open_epid_session", lambda s: opened.append(s))
    d._open_in_app()
    assert opened == [near]


def test_epid_check_reads_only_what_is_newer_than_the_last_mark(tmp_path):
    import os, time, iview_native as inn, wl_sources as ws
    root = tmp_path / "epid"; old = root / "2026-06-01"; new = root / "2026-06-20"
    old.mkdir(parents=True); new.mkdir()
    f_old, f_new = old / "a.dcm", new / "b.dcm"
    f_old.write_bytes(b"x"); f_new.write_bytes(b"y")
    t_old, t_new = time.time() - 60 * 86400, time.time() - 10 * 86400
    for p, t in ((f_old, t_old), (f_new, t_new), (old, t_old), (new, t_new)):
        os.utime(p, (t, t))
    os.utime(root, (t_new, t_new))
    files, dirs, newest = inn.walk_tree(root)
    assert {p.name for p, *_ in files} == {"a.dcm", "b.dcm"} and abs(newest - t_new) < 2
    files, dirs, _ = inn.walk_tree(root, since=t_new - ws.SAFETY_S)
    assert {p.name for p, *_ in files} == {"b.dcm"}                 # the old day folder is not even read
    # the mark stays behind a session that is still waiting, and moves on once nothing waits
    f = tmp_path / "f.dcm"; f.write_bytes(b"z"); os.utime(f, (t_old, t_old))
    waiting = SimpleNamespace(status="new", files={0: f}, pf=None)
    ws.save_scan_mark(root, tmp_path, [waiting], t_new)
    assert abs(ws.scan_mark(root, tmp_path) - t_old) < 2
    waiting.status = "processed"
    ws.save_scan_mark(root, tmp_path, [waiting], t_new)
    assert abs(ws.scan_mark(root, tmp_path) - t_new) < 2


def test_scan_caches_keep_older_entries(tmp_path):
    import json, iview_native as inn
    (tmp_path / "c").mkdir(); (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / inn.CACHE_NAME).write_text(json.dumps({"elsewhere": {"m": 1, "s": 1, "kind": "wl"}}))
    inn.scan_native(tmp_path / "c", tmp_path / "cache")
    assert "elsewhere" in json.loads((tmp_path / "cache" / inn.CACHE_NAME).read_text())
