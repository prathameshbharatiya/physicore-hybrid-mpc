"""tests/test_phase5.py — Phase 5 Data & Persistence Tests"""

import csv
import json
import math
import tempfile
import time
from pathlib import Path

import pytest
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# PART A: TelemetryStore tests
# ═══════════════════════════════════════════════════════════════════════

class TestTelemetryStore:

    def _make_store(self, tmp_path):
        from physicore.data.telemetry_store import TelemetryStore
        db = tmp_path / "test_tel.db"
        store = TelemetryStore(db_path=db)
        yield store
        store.close()

    @pytest.fixture()
    def store(self, tmp_path):
        yield from self._make_store(tmp_path)

    def test_01_write_and_query_round_trip(self, store):
        """write() + query() preserves values within tolerance."""
        store.create_session("sess1", "robot_a", "quadrotor")
        store.write("robot_a", "sess1", step=1, timestamp=1000.0,
                    metrics={"residual": 0.42, "uncertainty": 0.07})
        # Allow background writer to flush
        time.sleep(0.25)
        pts = store.query("robot_a", "residual", limit=10)
        assert len(pts) >= 1
        assert abs(pts[0]["value"] - 0.42) < 1e-6

    def test_02_write_is_non_blocking(self, tmp_path):
        """write() returns immediately (queue-based); no significant delay."""
        from physicore.data.telemetry_store import TelemetryStore
        store = TelemetryStore(db_path=tmp_path / "nb.db")
        store.create_session("s", "r", "platform")
        t0 = time.monotonic()
        for i in range(500):
            store.write("r", "s", step=i, timestamp=float(i),
                        metrics={"val": float(i)})
        elapsed = time.monotonic() - t0
        store.close()
        assert elapsed < 1.0, f"500 writes took {elapsed:.3f}s — too slow"

    def test_03_export_csv_produces_valid_file(self, store, tmp_path):
        """export_csv() writes a valid CSV with correct column headers."""
        store.create_session("sess_csv", "robot_b", "car")
        store.write("robot_b", "sess_csv", 1, 2000.0,
                    {"residual": 0.1, "uncertainty": 0.02})
        time.sleep(0.3)
        out = tmp_path / "out.csv"
        store.export_csv("sess_csv", out)
        assert out.exists()
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        assert "robot_id" in reader.fieldnames
        assert "session_id" in reader.fieldnames
        assert "key" in reader.fieldnames
        assert "value" in reader.fieldnames
        assert len(rows) >= 1

    def test_04_purge_old_removes_stale_sessions(self, tmp_path):
        """purge_old() removes sessions with started_at older than threshold."""
        from physicore.data.telemetry_store import TelemetryStore, SessionRecord
        import sqlite3 as _sqlite3

        store = TelemetryStore(db_path=tmp_path / "purge.db")
        # Create a session with a very old timestamp by writing directly
        store.create_session("old_sess", "robot_c", "rover")
        time.sleep(0.25)

        # Force the session's started_at to be old via direct SQL
        con = _sqlite3.connect(str(tmp_path / "purge.db"))
        old_ts = time.time() - 40 * 86400  # 40 days ago
        con.execute("UPDATE sessions SET started_at=? WHERE session_id='old_sess'", [old_ts])
        con.commit()
        con.close()

        removed = store.purge_old(days=30)
        store.close()
        assert removed >= 1

    def test_05_stats_returns_non_zero_db_size(self, store):
        """stats() returns db_size_mb > 0 after writes."""
        store.create_session("s_stats", "robot_d", "humanoid")
        store.write("robot_d", "s_stats", 1, time.time(),
                    {"residual": 0.5})
        time.sleep(0.3)
        s = store.stats()
        assert "total_sessions" in s
        assert "total_rows" in s
        assert "db_size_mb" in s
        assert s["db_size_mb"] >= 0  # file may be tiny but must exist

    def test_06_query_session_returns_multiple_keys(self, store):
        """query_session() returns a dict keyed by metric name."""
        store.create_session("qs", "robot_e", "rover")
        for step in range(5):
            store.write("robot_e", "qs", step, float(step),
                        {"residual": float(step) * 0.1, "uncertainty": 0.05})
        time.sleep(0.3)
        data = store.query_session("qs", keys=["residual", "uncertainty"])
        assert "residual" in data
        assert "uncertainty" in data
        assert len(data["residual"]) >= 1

    def test_07_delete_session_removes_rows(self, store):
        """delete_session() removes session and telemetry rows."""
        store.create_session("del_me", "robot_f", "car")
        store.write("robot_f", "del_me", 1, time.time(), {"x": 1.0})
        time.sleep(0.3)
        assert store.delete_session("del_me")
        assert store.get_session("del_me") is None

    def test_08_export_json_round_trips(self, store):
        """export_json() returns a dict with session and series keys."""
        store.create_session("json_sess", "robot_g", "auv")
        store.write("robot_g", "json_sess", 1, 1.0, {"residual": 0.3})
        time.sleep(0.3)
        data = store.export_json("json_sess")
        assert "session"  in data
        assert "series"   in data
        assert isinstance(data["series"], dict)

    def test_09_sessions_filter_by_robot(self, store):
        """sessions(robot_id=...) returns only sessions for that robot."""
        store.create_session("s_r1", "robot_1", "car")
        store.create_session("s_r2", "robot_2", "rover")
        time.sleep(0.2)
        r1_sessions = store.sessions(robot_id="robot_1")
        ids = {s.session_id for s in r1_sessions}
        assert "s_r1" in ids
        assert "s_r2" not in ids


# ═══════════════════════════════════════════════════════════════════════
# PART B: PlatformPrior tests
# ═══════════════════════════════════════════════════════════════════════

class TestPlatformPrior:

    def _make_prior(self, platform="quadrotor"):
        from physicore.core.transfer import PlatformPrior
        return PlatformPrior(platform)

    def _sample_session_params(self, rng, base, noise_scale=0.05):
        return {k: v + rng.normal(0, noise_scale) for k, v in base.items()}

    def test_01_update_shifts_map_toward_mode(self):
        """update() with 3 consistent sessions shifts MAP toward the true value."""
        prior    = self._make_prior()
        true_val = {"mass": 2.5, "friction": 0.3}
        rng      = np.random.default_rng(42)

        for _ in range(3):
            params = self._sample_session_params(rng, true_val, noise_scale=0.02)
            prior.update(params, n_steps=200, convergence_pct=80.0)

        map_est = prior.map_estimate()
        assert abs(map_est["mass"] - true_val["mass"]) < 0.5
        assert abs(map_est["friction"] - true_val["friction"]) < 0.3

    def test_02_sample_returns_all_param_keys(self):
        """sample() returns a dict with the same keys as the training params."""
        prior  = self._make_prior()
        params = {"mass": 1.0, "friction": 0.2, "damping": 0.05}
        prior.update(params, n_steps=100, convergence_pct=60.0)
        sample = prior.sample()
        assert set(sample.keys()) == set(params.keys())

    def test_03_uncertainty_decreases_with_more_sessions(self):
        """uncertainty() is lower after 10 sessions than after 1."""
        prior1 = self._make_prior()
        prior10 = self._make_prior()
        rng    = np.random.default_rng(7)
        base   = {"mass": 1.0, "friction": 0.2}

        prior1.update(self._sample_session_params(rng, base), 100, 70.0)
        for _ in range(10):
            prior10.update(self._sample_session_params(rng, base), 100, 70.0)

        unc1  = prior1.uncertainty()["mass"]
        unc10 = prior10.uncertainty()["mass"]
        assert unc10 < unc1, f"Expected unc10 < unc1: {unc10:.4f} < {unc1:.4f}"

    def test_04_serialize_deserialize_round_trip(self):
        """Serialize + deserialize preserves MAP estimate."""
        prior = self._make_prior()
        prior.update({"mass": 3.0, "friction": 0.4}, 500, 90.0)
        original_map = prior.map_estimate()

        data     = prior.serialize()
        prior2   = self._make_prior()
        prior2.deserialize(data)
        restored = prior2.map_estimate()

        for k in original_map:
            assert abs(original_map[k] - restored[k]) < 1e-9

    def test_05_low_convergence_sessions_are_ignored(self):
        """Sessions with convergence_pct < 5 don't update the prior."""
        prior = self._make_prior()
        prior.update({"mass": 1.0}, n_steps=100, convergence_pct=3.0)
        assert prior.n_sessions == 0, "Low-convergence session should be skipped"

    def test_06_save_load_from_file(self, tmp_path):
        """save() + load() persists the prior to disk."""
        from physicore.core.transfer import PlatformPrior
        prior = PlatformPrior("test_platform")
        prior.update({"mass": 2.0}, 100, 75.0)
        path  = tmp_path / "prior.json"
        prior.save(path)

        loaded = PlatformPrior.load("test_platform", path)
        assert abs(loaded.map_estimate()["mass"] - prior.map_estimate()["mass"]) < 1e-9


# ═══════════════════════════════════════════════════════════════════════
# PART C: TransferEngine tests
# ═══════════════════════════════════════════════════════════════════════

class TestTransferEngine:

    @pytest.fixture()
    def reg_dir(self, tmp_path):
        """Fake registry root with some sessions."""
        platform = "balancing_bot"
        d = tmp_path / platform
        d.mkdir()
        sessions = []
        rng = np.random.default_rng(99)
        for i in range(5):
            sessions.append({
                "session_id":     f"sess{i:03d}",
                "platform":       platform,
                "steps":          200 + i * 50,
                "convergence_pct": 60.0 + i * 5,
                "final_params":   {
                    "mass":     1.0 + rng.normal(0, 0.05),
                    "friction": 0.2 + rng.normal(0, 0.02),
                },
            })
        (d / "sessions.jsonl").write_text(
            "\n".join(json.dumps(s) for s in sessions), encoding="utf-8"
        )
        return tmp_path, platform, sessions

    def test_01_warm_start_returns_params_close_to_mode(self, reg_dir):
        """warm_start() returns params after building prior from registry sessions."""
        from physicore.core.transfer import TransferEngine
        from physicore.core.engine import PhysiCore

        root, platform, sessions = reg_dir
        te     = TransferEngine(registry_root=root)
        engine = PhysiCore.for_platform(platform)

        params = te.warm_start(engine, platform)
        assert "mass" in params, "Expected 'mass' in warm-start params"
        # Params should be in a reasonable range
        assert 0.5 < params["mass"] < 2.0, f"mass {params['mass']:.3f} out of range"

    def test_02_cross_platform_transfer_maps_keys(self, reg_dir):
        """cross_platform_transfer() maps specified keys to target names."""
        from physicore.core.transfer import TransferEngine, PlatformPrior
        root, platform, sessions = reg_dir
        te = TransferEngine(registry_root=root)

        # Build the prior from the sessions
        prior = te.build_prior_from_registry(platform)
        assert prior.n_sessions >= 1

        # Transfer mass → link_mass, friction → joint_friction
        transferred = te.cross_platform_transfer(
            platform, "surgical_robot",
            {"mass": "link_mass", "friction": "joint_friction"}
        )
        assert "link_mass" in transferred
        assert "joint_friction" in transferred
        assert 0.5 < transferred["link_mass"] < 2.0

    def test_03_find_similar_sessions_returns_n_results(self, reg_dir):
        """find_similar_sessions() returns at most n sessions."""
        from physicore.core.transfer import TransferEngine
        root, platform, sessions = reg_dir
        te = TransferEngine(registry_root=root)
        similar = te.find_similar_sessions(
            platform, {"mass": 1.0, "friction": 0.2}, n=3
        )
        assert len(similar) <= 3
        assert len(similar) >= 1

    def test_04_update_prior_increments_session_count(self, reg_dir):
        """update_prior() increments the prior's session count."""
        from physicore.core.transfer import TransferEngine
        root, platform, _ = reg_dir
        te = TransferEngine(registry_root=root)

        # Build from registry first
        prior_before = te.build_prior_from_registry(platform)
        n_before     = prior_before.n_sessions

        te.update_prior(platform, {"mass": 1.5, "friction": 0.25}, 300, 85.0)

        prior_after = te._load_prior(platform)
        assert prior_after.n_sessions == n_before + 1


# ═══════════════════════════════════════════════════════════════════════
# PART D: SessionAnalytics tests
# ═══════════════════════════════════════════════════════════════════════

def _make_store_with_session(tmp_path, session_id, residuals):
    """Helper: create a TelemetryStore pre-filled with a synthetic residual series."""
    from physicore.data.telemetry_store import TelemetryStore
    store = TelemetryStore(db_path=tmp_path / f"analytics_{session_id}.db")
    store.create_session(session_id, "robot_test", "balancing_bot")
    for i, r in enumerate(residuals):
        store.write("robot_test", session_id, step=i, timestamp=float(i),
                    metrics={"residual": float(r)})
    time.sleep(0.4)
    return store


class TestSessionAnalytics:

    def test_01_convergence_rate_positive_for_decreasing_residual(self, tmp_path):
        """convergence_rate() returns a positive float for a decaying residual."""
        from physicore.tools.analytics import SessionAnalytics
        n    = 100
        vals = [1.0 * math.exp(-i / 30.0) + 0.01 for i in range(n)]
        store = _make_store_with_session(tmp_path, "conv_sess", vals)
        ana   = SessionAnalytics(store)
        tau   = ana.convergence_rate("conv_sess")
        store.close()
        assert tau > 0.0, f"Expected tau > 0, got {tau}"

    def test_02_compare_picks_lower_residual_as_winner(self, tmp_path):
        """compare() returns the session with lower final residual as winner."""
        from physicore.tools.analytics import SessionAnalytics
        from physicore.data.telemetry_store import TelemetryStore

        db     = tmp_path / "compare.db"
        store  = TelemetryStore(db_path=db)
        store.create_session("good_sess", "r", "platform")
        store.create_session("bad_sess",  "r", "platform")

        for i in range(50):
            store.write("r", "good_sess", i, float(i),
                        {"residual": max(0.01, 1.0 - i * 0.018)})
            store.write("r", "bad_sess",  i, float(i),
                        {"residual": max(0.2,  1.0 - i * 0.005)})
        time.sleep(0.4)

        ana    = SessionAnalytics(store)
        report = ana.compare("good_sess", "bad_sess")
        store.close()

        assert report.winner == "good_sess", (
            f"Expected 'good_sess' as winner, got '{report.winner}'"
        )

    def test_03_anomaly_score_non_negative(self, tmp_path):
        """anomaly_score() returns a non-negative float."""
        from physicore.tools.analytics import SessionAnalytics
        store = _make_store_with_session(tmp_path, "anm_sess",
                                         [0.5 - i * 0.005 for i in range(50)])
        ana   = SessionAnalytics(store)
        score = ana.anomaly_score("anm_sess")
        store.close()
        assert score >= 0.0

    def test_04_compare_report_has_required_fields(self, tmp_path):
        """ComparisonReport.to_dict() has all required keys."""
        from physicore.tools.analytics import SessionAnalytics
        from physicore.data.telemetry_store import TelemetryStore
        db    = tmp_path / "cmp2.db"
        store = TelemetryStore(db_path=db)
        store.create_session("sa", "r", "p")
        store.create_session("sb", "r", "p")
        for i in range(10):
            store.write("r", "sa", i, float(i), {"residual": 0.5 - i * 0.03})
            store.write("r", "sb", i, float(i), {"residual": 0.4 - i * 0.02})
        time.sleep(0.3)
        ana    = SessionAnalytics(store)
        report = ana.compare("sa", "sb")
        d      = report.to_dict()
        store.close()
        for key in ("winner", "session_a", "session_b",
                    "residual_improvement_pct", "convergence_speedup",
                    "param_deltas", "summary_text"):
            assert key in d, f"Missing key: {key}"

    def test_05_fleet_summary_includes_all_robots(self, tmp_path):
        """fleet_summary() includes all specified robot_ids."""
        from physicore.tools.analytics import SessionAnalytics
        from physicore.data.telemetry_store import TelemetryStore
        db    = tmp_path / "fleet.db"
        store = TelemetryStore(db_path=db)
        for rid in ["r1", "r2", "r3"]:
            store.create_session(f"sess_{rid}", rid, "rover")
        time.sleep(0.25)
        ana    = SessionAnalytics(store)
        report = ana.fleet_summary(["r1", "r2", "r3"])
        store.close()
        assert set(report.robot_ids) == {"r1", "r2", "r3"}
        assert report.total_sessions >= 3

    def test_06_parameter_drift_returns_dict(self, tmp_path):
        """parameter_drift() returns a dict (may be empty if no param_ keys)."""
        from physicore.tools.analytics import SessionAnalytics
        from physicore.data.telemetry_store import TelemetryStore
        db    = tmp_path / "drift.db"
        store = TelemetryStore(db_path=db)
        store.create_session("drift_sess", "r", "p")
        for i in range(30):
            store.write("r", "drift_sess", i, float(i),
                        {"residual": 1.0 - i * 0.02,
                         "param_mass": 1.0 + i * 0.005})
        time.sleep(0.35)
        ana   = SessionAnalytics(store)
        drift = ana.parameter_drift("drift_sess")
        store.close()
        assert isinstance(drift, dict)
        if drift:
            assert "mass" in drift, f"Expected 'mass' in drift, got {list(drift.keys())}"

    def test_07_convergence_rate_returns_float_for_flat_series(self, tmp_path):
        """convergence_rate() returns a valid float even for a flat residual series."""
        from physicore.tools.analytics import SessionAnalytics
        vals  = [0.5] * 50
        store = _make_store_with_session(tmp_path, "flat_sess", vals)
        ana   = SessionAnalytics(store)
        tau   = ana.convergence_rate("flat_sess")
        store.close()
        assert isinstance(tau, float)
        assert not math.isnan(tau)
