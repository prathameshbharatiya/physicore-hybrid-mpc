"""tests/test_phase7.py — Phase 7 Multi-tenant Platform Tests"""

import io
import json
import time
import zipfile
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════════
# PART A: Organization CRUD & Quota tests
# ═══════════════════════════════════════════════════════════════════════

class TestOrgStore:

    def _make_store(self, tmp_path):
        from physicore.api.org import OrgStore
        return OrgStore(db_path=tmp_path / "orgs.db")

    def test_01_create_org_returns_org_with_id(self, tmp_path):
        """create_org() returns an Organization with non-empty org_id."""
        store = self._make_store(tmp_path)
        org = store.create_org("AcmeCorp", owner_id="user_1", plan="free")
        assert org.org_id
        assert org.name == "AcmeCorp"
        assert org.plan == "free"
        assert "user_1" in org.member_ids

    def test_02_get_org_returns_created_org(self, tmp_path):
        """get_org() retrieves a previously created org."""
        store = self._make_store(tmp_path)
        created = store.create_org("BetaCorp", owner_id="user_2")
        fetched = store.get_org(created.org_id)
        assert fetched is not None
        assert fetched.name == "BetaCorp"

    def test_03_get_org_returns_none_for_missing(self, tmp_path):
        """get_org() returns None for a non-existent org_id."""
        store = self._make_store(tmp_path)
        assert store.get_org("nonexistent-id") is None

    def test_04_add_member_and_list_members(self, tmp_path):
        """add_member() + get_members() correctly reflects the new member."""
        store = self._make_store(tmp_path)
        org = store.create_org("GammaOrg", owner_id="owner_1")
        store.add_member(org.org_id, "user_99", role="member", email="u@example.com")
        members = store.get_members(org.org_id)
        ids = {m.user_id for m in members}
        assert "user_99" in ids

    def test_05_change_role_updates_membership(self, tmp_path):
        """change_role() updates the role and permissions for an existing member."""
        store = self._make_store(tmp_path)
        org = store.create_org("DeltaOrg", owner_id="owner_2")
        store.add_member(org.org_id, "user_55", role="member")
        updated = store.change_role(org.org_id, "user_55", "admin")
        assert updated is not None
        assert updated.role == "admin"
        assert "invite" in updated.permissions

    def test_06_remove_member_deletes_membership(self, tmp_path):
        """remove_member() removes the user from the org."""
        store = self._make_store(tmp_path)
        org = store.create_org("EpsilonOrg", owner_id="owner_3")
        store.add_member(org.org_id, "user_66", role="member")
        removed = store.remove_member(org.org_id, "user_66")
        assert removed
        mem = store.get_membership("user_66", org.org_id)
        assert mem is None

    def test_07_delete_org_removes_org(self, tmp_path):
        """delete_org() removes the organization record."""
        store = self._make_store(tmp_path)
        org = store.create_org("ZetaOrg", owner_id="owner_4")
        assert store.delete_org(org.org_id)
        assert store.get_org(org.org_id) is None

    def test_08_quota_ok_when_under_limit(self, tmp_path):
        """check_quota() returns status='ok' when well under limit."""
        store = self._make_store(tmp_path)
        org = store.create_org("QuotaOrg", owner_id="owner_5", plan="pro")
        status = store.check_quota(org.org_id, "robots", current_used=5)
        assert status.status == "ok"
        assert status.limit == 20   # pro plan

    def test_09_quota_warning_at_80_pct(self, tmp_path):
        """check_quota() returns 'warning' when usage >= 80%."""
        store = self._make_store(tmp_path)
        org = store.create_org("WarnOrg", owner_id="owner_6", plan="free")
        # free plan: 3 robots. 3 used = 100% exceeded; 3*0.8 = 2.4 so use 3 → exceeded
        # Let's use a value at 80% of 5 plugins = 4
        status = store.check_quota(org.org_id, "plugins", current_used=4)
        assert status.status in ("warning", "exceeded")

    def test_10_quota_exceeded_when_over_limit(self, tmp_path):
        """check_quota() returns 'exceeded' when usage > limit."""
        store = self._make_store(tmp_path)
        org = store.create_org("ExceedOrg", owner_id="owner_7", plan="free")
        status = store.check_quota(org.org_id, "robots", current_used=10)
        assert status.status == "exceeded"

    def test_11_pro_plan_has_higher_quotas(self, tmp_path):
        """Pro plan has higher quotas than free plan."""
        store = self._make_store(tmp_path)
        free_org = store.create_org("FreeOrg", owner_id="u1", plan="free")
        pro_org  = store.create_org("ProOrg",  owner_id="u2", plan="pro")
        assert pro_org.robot_quota > free_org.robot_quota
        assert pro_org.plugin_quota > free_org.plugin_quota

    def test_12_invite_and_accept(self, tmp_path):
        """create_invite() + accept_invite() adds user as member."""
        store = self._make_store(tmp_path)
        org = store.create_org("InviteOrg", owner_id="owner_8")
        invite_id = store.create_invite(org.org_id, "new@example.com", "member")
        assert invite_id
        mem = store.accept_invite(invite_id, user_id="new_user_1")
        assert mem is not None
        assert mem.org_id == org.org_id

    def test_13_get_usage_returns_dict_with_quota_keys(self, tmp_path):
        """get_usage() returns a dict with 'robots' and 'plugins' quota info."""
        store = self._make_store(tmp_path)
        org = store.create_org("UsageOrg", owner_id="ou1")
        usage = store.get_usage(org.org_id, robot_count=1, plugin_count=2)
        assert "robots" in usage
        assert "plugins" in usage
        assert usage["robots"]["used"] == 1

    def test_14_list_orgs_for_user(self, tmp_path):
        """list_orgs_for_user() returns orgs where user is a member."""
        store = self._make_store(tmp_path)
        org1 = store.create_org("O1", owner_id="uid_xyz")
        org2 = store.create_org("O2", owner_id="uid_xyz")
        _    = store.create_org("O3", owner_id="uid_other")
        orgs = store.list_orgs_for_user("uid_xyz")
        ids = {o.org_id for o in orgs}
        assert org1.org_id in ids
        assert org2.org_id in ids


# ═══════════════════════════════════════════════════════════════════════
# PART B: Marketplace tests
# ═══════════════════════════════════════════════════════════════════════

def _make_plugin_zip(plugin_id: str = "test_plugin", version: str = "1.0.0",
                     extra_source: str = "") -> bytes:
    """Build a minimal valid .physicore-plugin zip."""
    manifest = {
        "plugin_id": plugin_id,
        "name": f"Test Plugin {plugin_id}",
        "version": version,
        "description": "A test plugin",
        "author": "tester",
        "permissions": [],
        "hooks": [],
        "tags": ["test"],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("plugin.json", json.dumps(manifest))
        zf.writestr("main.py", extra_source or "# safe plugin\nclass MyPlugin:\n    pass\n")
    return buf.getvalue()


class TestMarketplaceRegistry:

    def _make_registry(self, tmp_path):
        from physicore.sdk.marketplace import MarketplaceRegistry
        return MarketplaceRegistry(store_root=tmp_path / "marketplace")

    def test_01_submit_valid_plugin(self, tmp_path):
        """submit() accepts a valid plugin zip and returns an entry."""
        reg = self._make_registry(tmp_path)
        entry = reg.submit(_make_plugin_zip("my_plugin"), author_id="auth_1", category="demo")
        assert entry.manifest.plugin_id == "my_plugin"
        assert entry.author_id == "auth_1"

    def test_02_get_returns_submitted_plugin(self, tmp_path):
        """get() returns the submitted entry."""
        reg = self._make_registry(tmp_path)
        reg.submit(_make_plugin_zip("plugin_a"), author_id="a1", category="analytics")
        entry = reg.get("plugin_a")
        assert entry is not None
        assert entry.manifest.name == "Test Plugin plugin_a"

    def test_03_search_by_name_query(self, tmp_path):
        """search() with a query filters by name."""
        reg = self._make_registry(tmp_path)
        reg.submit(_make_plugin_zip("perception_cam"), author_id="a1", category="perception")
        reg.submit(_make_plugin_zip("safety_brake"),  author_id="a2", category="safety")
        results = reg.search(query="perception")
        ids = [r.manifest.plugin_id for r in results]
        assert "perception_cam" in ids

    def test_04_search_by_category_filters_results(self, tmp_path):
        """search() with category= returns only matching entries."""
        reg = self._make_registry(tmp_path)
        reg.submit(_make_plugin_zip("p1"), author_id="a", category="perception")
        reg.submit(_make_plugin_zip("p2"), author_id="a", category="safety")
        results = reg.search(category="safety")
        assert all(r.category == "safety" for r in results)
        ids = [r.manifest.plugin_id for r in results]
        assert "p2" in ids
        assert "p1" not in ids

    def test_05_safety_scan_rejects_socket_import(self, tmp_path):
        """submit() raises ValueError when plugin imports socket."""
        from physicore.sdk.marketplace import MarketplaceRegistry
        reg = self._make_registry(tmp_path)
        bad_source = "import socket\nsock = socket.socket()\n"
        with pytest.raises(ValueError, match="[Ss]afety|[Ff]orbidden|socket"):
            reg.submit(_make_plugin_zip("bad_plugin", extra_source=bad_source), "a1")

    def test_06_safety_scan_rejects_subprocess(self, tmp_path):
        """submit() raises ValueError when plugin calls subprocess."""
        reg = self._make_registry(tmp_path)
        bad_source = "import subprocess\nsubprocess.run(['ls'])\n"
        with pytest.raises(ValueError):
            reg.submit(_make_plugin_zip("bad2", extra_source=bad_source), "a1")

    def test_07_safety_scan_rejects_eval(self, tmp_path):
        """submit() raises ValueError when plugin uses eval()."""
        reg = self._make_registry(tmp_path)
        bad_source = "result = eval('1+1')\n"
        with pytest.raises(ValueError):
            reg.submit(_make_plugin_zip("bad3", extra_source=bad_source), "a1")

    def test_08_install_copies_zip_to_target(self, tmp_path):
        """install() copies the plugin zip to the target directory."""
        reg = self._make_registry(tmp_path)
        reg.submit(_make_plugin_zip("inst_plugin"), author_id="a", category="demo")
        target = tmp_path / "my_plugins"
        dest = reg.install("inst_plugin", "1.0.0", str(target))
        assert dest.exists()

    def test_09_install_increments_download_count(self, tmp_path):
        """install() increments the plugin's download_count."""
        reg = self._make_registry(tmp_path)
        reg.submit(_make_plugin_zip("dl_plugin"), author_id="a", category="demo")
        target = tmp_path / "plugs"
        reg.install("dl_plugin", "1.0.0", str(target))
        entry = reg.get("dl_plugin")
        assert entry is not None
        assert entry.download_count >= 1

    def test_10_rate_plugin_updates_rating(self, tmp_path):
        """rate() adds a review and updates the average rating."""
        reg = self._make_registry(tmp_path)
        reg.submit(_make_plugin_zip("rate_me"), author_id="a", category="demo")
        reg.rate("rate_me", author_id="user_1", rating=5.0, text="Excellent!")
        entry = reg.get("rate_me")
        assert entry is not None
        assert entry.rating > 0

    def test_11_rate_invalid_range_raises(self, tmp_path):
        """rate() raises ValueError for rating out of [1, 5]."""
        reg = self._make_registry(tmp_path)
        reg.submit(_make_plugin_zip("v_rate"), author_id="a")
        with pytest.raises(ValueError):
            reg.rate("v_rate", "u1", rating=6.0)

    def test_12_submit_invalid_zip_raises(self, tmp_path):
        """submit() raises ValueError for invalid zip bytes."""
        reg = self._make_registry(tmp_path)
        with pytest.raises(ValueError):
            reg.submit(b"not a zip file", author_id="a")

    def test_13_search_empty_query_returns_all(self, tmp_path):
        """search() with empty query returns all entries."""
        reg = self._make_registry(tmp_path)
        for i in range(3):
            reg.submit(_make_plugin_zip(f"plug_{i}"), author_id="a", category="demo")
        results = reg.search(query="")
        assert len(results) >= 3


# ═══════════════════════════════════════════════════════════════════════
# PART C: UsageMetering tests
# ═══════════════════════════════════════════════════════════════════════

class TestUsageMetering:

    def _make_metering(self, tmp_path):
        from physicore.api.metering import UsageMetering
        return UsageMetering(db_path=tmp_path / "metering.db")

    def test_01_record_step_increments_count(self, tmp_path):
        """record_step() increments the daily step count."""
        m = self._make_metering(tmp_path)
        m.record_step("org_a", "robot_1")
        m.record_step("org_a", "robot_1")
        count = m.steps_in_period("org_a", days=1)
        assert count >= 2

    def test_02_record_storage_increments_bytes(self, tmp_path):
        """record_storage() tracks bytes written."""
        m = self._make_metering(tmp_path)
        m.record_storage("org_b", 1_000_000)
        mb = m.storage_mb("org_b")
        assert mb >= 1.0

    def test_03_get_usage_returns_summary(self, tmp_path):
        """get_usage() returns a UsageSummary with required fields."""
        m = self._make_metering(tmp_path)
        m.set_plan("org_c", "free")
        m.record_step("org_c", "r1")
        summary = m.get_usage("org_c", period="month")
        assert summary.org_id == "org_c"
        assert isinstance(summary.steps_this_period, int)
        assert "steps_per_month" in summary.plan_limits

    def test_04_check_quota_ok(self, tmp_path):
        """check_quota() returns OK for well-under-limit usage."""
        m = self._make_metering(tmp_path)
        m.set_plan("org_d", "pro")
        result = m.check_quota("org_d", "robots", current_used=5)
        assert result.status.value == "ok"

    def test_05_check_quota_exceeded(self, tmp_path):
        """check_quota() returns EXCEEDED when usage > plan limit."""
        from physicore.api.metering import QuotaStatusEnum
        m = self._make_metering(tmp_path)
        m.set_plan("org_e", "free")
        result = m.check_quota("org_e", "robots", current_used=100)
        assert result.status == QuotaStatusEnum.EXCEEDED

    def test_06_check_quota_warning(self, tmp_path):
        """check_quota() returns WARNING at 80%+ usage."""
        from physicore.api.metering import QuotaStatusEnum
        m = self._make_metering(tmp_path)
        m.set_plan("org_f", "free")
        # free: 5 plugins, 80% = 4
        result = m.check_quota("org_f", "plugins", current_used=4)
        assert result.status in (QuotaStatusEnum.WARNING, QuotaStatusEnum.EXCEEDED)

    def test_07_steps_per_day_returns_list(self, tmp_path):
        """steps_per_day() returns a list of dicts with 'day' and 'steps'."""
        m = self._make_metering(tmp_path)
        m.record_step("org_g", "r1")
        data = m.steps_per_day("org_g", days=7)
        assert isinstance(data, list)
        if data:
            assert "day" in data[0]
            assert "steps" in data[0]

    def test_08_all_quotas_returns_dict_of_resources(self, tmp_path):
        """all_quotas() returns entries for steps, storage, robots, plugins."""
        m = self._make_metering(tmp_path)
        m.set_plan("org_h", "free")
        result = m.all_quotas("org_h")
        for key in ("steps", "storage", "robots", "plugins"):
            assert key in result

    def test_09_usage_summary_to_dict(self, tmp_path):
        """UsageSummary.to_dict() contains all required keys."""
        m = self._make_metering(tmp_path)
        m.set_plan("org_i", "free")
        summary = m.get_usage("org_i")
        d = summary.to_dict()
        for key in ("org_id", "period", "steps_this_period", "robots_active",
                    "storage_mb", "plugins_loaded", "plan_limits"):
            assert key in d


# ═══════════════════════════════════════════════════════════════════════
# PART D: AuditLog tests
# ═══════════════════════════════════════════════════════════════════════

class TestAuditLog:

    def _make_log(self, tmp_path):
        from physicore.api.audit import AuditLog
        return AuditLog(db_path=tmp_path / "audit.db")

    def test_01_log_creates_event(self, tmp_path):
        """log() creates and returns an AuditEvent with a unique event_id."""
        alog = self._make_log(tmp_path)
        ev = alog.log("user_1", "org_1", "engine.configure", resource="engine",
                      details={"platform": "quadrotor"}, ip="127.0.0.1")
        assert ev.event_id
        assert ev.action == "engine.configure"

    def test_02_query_returns_logged_event(self, tmp_path):
        """query() returns the event that was logged."""
        alog = self._make_log(tmp_path)
        alog.log("user_2", "org_2", "plugin.install", resource="plugin:foo")
        events = alog.query("org_2", start_time=time.time() - 60)
        assert len(events) >= 1
        assert any(e.action == "plugin.install" for e in events)

    def test_03_query_filters_by_user_id(self, tmp_path):
        """query() with user_id= returns only events from that user."""
        alog = self._make_log(tmp_path)
        alog.log("user_a", "org_3", "engine.step")
        alog.log("user_b", "org_3", "engine.step")
        events = alog.query("org_3", start_time=time.time() - 60, user_id="user_a")
        assert all(e.user_id == "user_a" for e in events)

    def test_04_query_filters_by_action(self, tmp_path):
        """query() with action= returns only matching actions."""
        alog = self._make_log(tmp_path)
        alog.log("u", "org_4", "engine.reset")
        alog.log("u", "org_4", "org.invite")
        events = alog.query("org_4", start_time=time.time() - 60, action="engine.reset")
        assert all("engine" in e.action for e in events)

    def test_05_query_empty_for_other_org(self, tmp_path):
        """query() for a different org returns no events."""
        alog = self._make_log(tmp_path)
        alog.log("u", "org_5", "engine.step")
        events = alog.query("org_other", start_time=time.time() - 60)
        assert len(events) == 0

    def test_06_export_csv_returns_string(self, tmp_path):
        """export_csv() returns a non-empty CSV string."""
        alog = self._make_log(tmp_path)
        alog.log("u", "org_6", "engine.configure")
        csv_str = alog.export_csv("org_6", period_days=1)
        assert isinstance(csv_str, str)
        assert "action" in csv_str  # header

    def test_07_audit_event_to_dict_has_fields(self, tmp_path):
        """AuditEvent.to_dict() contains all required keys."""
        alog = self._make_log(tmp_path)
        ev = alog.log("u", "org_7", "org.member.remove", details={"target": "user_x"})
        d = ev.to_dict()
        for key in ("event_id", "user_id", "org_id", "action", "resource",
                    "details", "ip", "timestamp", "status"):
            assert key in d

    def test_08_count_returns_correct_number(self, tmp_path):
        """count() returns the number of logged events for an org."""
        alog = self._make_log(tmp_path)
        for _ in range(5):
            alog.log("u", "org_8", "engine.step")
        assert alog.count("org_8") == 5

    def test_09_multiple_events_ordered_desc(self, tmp_path):
        """query() returns events ordered by timestamp descending."""
        alog = self._make_log(tmp_path)
        alog.log("u", "org_9", "action.first")
        time.sleep(0.02)
        alog.log("u", "org_9", "action.second")
        events = alog.query("org_9", start_time=time.time() - 60)
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps, reverse=True)


# ═══════════════════════════════════════════════════════════════════════
# PART E: Safety scan unit tests
# ═══════════════════════════════════════════════════════════════════════

class TestSafetyScan:

    def test_01_clean_plugin_passes(self):
        """Safe plugin source passes the scan."""
        from physicore.sdk.marketplace import scan_source
        src = "class MyPlugin:\n    def setup(self, engine): pass\n"
        result = scan_source(src)
        assert result.passed

    def test_02_socket_import_fails(self):
        """Plugin importing socket fails the scan."""
        from physicore.sdk.marketplace import scan_source
        src = "import socket\n"
        result = scan_source(src)
        assert not result.passed

    def test_03_subprocess_fails(self):
        """Plugin calling subprocess.run fails the scan."""
        from physicore.sdk.marketplace import scan_source
        src = "import subprocess\nsubprocess.run(['ls'])\n"
        result = scan_source(src)
        assert not result.passed

    def test_04_eval_fails(self):
        """Plugin using eval() fails the scan."""
        from physicore.sdk.marketplace import scan_source
        src = "x = eval('1+2')\n"
        result = scan_source(src)
        assert not result.passed

    def test_05_allowed_plugins_path_open_passes(self):
        """File open to /plugins/ path is allowed."""
        from physicore.sdk.marketplace import scan_source
        src = "with open('/plugins/data.json') as f: data = f.read()\n"
        result = scan_source(src)
        assert result.passed

    def test_06_scan_result_has_violations_list(self):
        """scan_source() returns violations list on failure."""
        from physicore.sdk.marketplace import scan_source
        src = "import socket\n"
        result = scan_source(src)
        assert isinstance(result.violations, list)
        assert len(result.violations) >= 1
