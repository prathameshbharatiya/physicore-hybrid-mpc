"""physicore/api/metering.py — Usage metering and billing hooks."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH: Path = Path.home() / ".physicore" / "metering.db"
_metering_instance: Optional["UsageMetering"] = None
_metering_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class QuotaStatusEnum(str, Enum):
    OK       = "ok"
    WARNING  = "warning"    # ≥ 80% used
    EXCEEDED = "exceeded"   # ≥ 100% used


@dataclass
class UsageSummary:
    org_id: str
    period: str
    steps_this_period: int
    robots_active: int
    storage_mb: float
    plugins_loaded: int
    plan_limits: Dict[str, Any]
    days_in_period: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "org_id": self.org_id,
            "period": self.period,
            "steps_this_period": self.steps_this_period,
            "robots_active": self.robots_active,
            "storage_mb": round(self.storage_mb, 3),
            "plugins_loaded": self.plugins_loaded,
            "plan_limits": self.plan_limits,
            "days_in_period": self.days_in_period,
        }


@dataclass
class QuotaCheckResult:
    resource: str
    used: int
    limit: int
    status: QuotaStatusEnum
    pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource,
            "used": self.used,
            "limit": self.limit,
            "status": self.status.value,
            "pct": round(self.pct, 1),
        }


_PLAN_LIMITS: Dict[str, Dict[str, Any]] = {
    "free": {
        "steps_per_month": 100_000,
        "storage_mb":       500,
        "robots":             3,
        "plugins":            5,
    },
    "pro": {
        "steps_per_month": 5_000_000,
        "storage_mb":      50_000,
        "robots":              20,
        "plugins":             50,
    },
    "enterprise": {
        "steps_per_month": 999_999_999,
        "storage_mb":      999_999_999,
        "robots":              500,
        "plugins":             500,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# UsageMetering
# ─────────────────────────────────────────────────────────────────────────────

class UsageMetering:
    """Records and queries per-org usage counters backed by SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path), check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_db(self) -> None:
        with self._conn() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS step_counts (
                    org_id TEXT NOT NULL,
                    day    TEXT NOT NULL,   -- ISO date YYYY-MM-DD
                    count  INTEGER DEFAULT 0,
                    PRIMARY KEY (org_id, day)
                );
                CREATE TABLE IF NOT EXISTS storage_usage (
                    org_id     TEXT PRIMARY KEY,
                    bytes_total INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS robot_activity (
                    org_id   TEXT NOT NULL,
                    robot_id TEXT NOT NULL,
                    last_seen REAL,
                    PRIMARY KEY (org_id, robot_id)
                );
                CREATE TABLE IF NOT EXISTS org_plans (
                    org_id TEXT PRIMARY KEY,
                    plan   TEXT DEFAULT 'free'
                );
            """)

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_step(self, org_id: str, robot_id: str, n: int = 1) -> None:
        day = _today()
        with self._lock, self._conn() as con:
            con.execute(
                "INSERT INTO step_counts (org_id, day, count) VALUES (?,?,?) "
                "ON CONFLICT(org_id, day) DO UPDATE SET count=count+?",
                (org_id, day, n, n),
            )
            con.execute(
                "INSERT INTO robot_activity VALUES (?,?,?) "
                "ON CONFLICT(org_id, robot_id) DO UPDATE SET last_seen=?",
                (org_id, robot_id, time.time(), time.time()),
            )

    def record_storage(self, org_id: str, bytes_written: int) -> None:
        with self._lock, self._conn() as con:
            con.execute(
                "INSERT INTO storage_usage (org_id, bytes_total) VALUES (?,?) "
                "ON CONFLICT(org_id) DO UPDATE SET bytes_total=bytes_total+?",
                (org_id, bytes_written, bytes_written),
            )

    def set_plan(self, org_id: str, plan: str) -> None:
        with self._lock, self._conn() as con:
            con.execute(
                "INSERT INTO org_plans (org_id, plan) VALUES (?,?) "
                "ON CONFLICT(org_id) DO UPDATE SET plan=?",
                (org_id, plan, plan),
            )

    # ── Querying ──────────────────────────────────────────────────────────────

    def _get_plan(self, org_id: str) -> str:
        with self._conn() as con:
            row = con.execute("SELECT plan FROM org_plans WHERE org_id=?", (org_id,)).fetchone()
        return row["plan"] if row else "free"

    def steps_in_period(self, org_id: str, days: int = 30) -> int:
        cutoff = _day_cutoff(days)
        with self._conn() as con:
            row = con.execute(
                "SELECT COALESCE(SUM(count), 0) as total FROM step_counts "
                "WHERE org_id=? AND day>=?",
                (org_id, cutoff),
            ).fetchone()
        return int(row[0]) if row else 0

    def steps_per_day(self, org_id: str, days: int = 30) -> List[Dict[str, Any]]:
        cutoff = _day_cutoff(days)
        with self._conn() as con:
            rows = con.execute(
                "SELECT day, count FROM step_counts WHERE org_id=? AND day>=? ORDER BY day",
                (org_id, cutoff),
            ).fetchall()
        return [{"day": r["day"], "steps": r["count"]} for r in rows]

    def active_robots(self, org_id: str, within_s: float = 3600.0) -> int:
        cutoff = time.time() - within_s
        with self._conn() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM robot_activity WHERE org_id=? AND last_seen>=?",
                (org_id, cutoff),
            ).fetchone()
        return int(row[0]) if row else 0

    def storage_mb(self, org_id: str) -> float:
        with self._conn() as con:
            row = con.execute(
                "SELECT bytes_total FROM storage_usage WHERE org_id=?", (org_id,)
            ).fetchone()
        return (row["bytes_total"] / 1e6) if row else 0.0

    def get_usage(self, org_id: str, period: str = "month",
                  robots_active: int = 0, plugins_loaded: int = 0) -> UsageSummary:
        days = {"month": 30, "week": 7, "day": 1}.get(period, 30)
        plan = self._get_plan(org_id)
        steps = self.steps_in_period(org_id, days)
        storage = self.storage_mb(org_id)
        if robots_active == 0:
            robots_active = self.active_robots(org_id)

        return UsageSummary(
            org_id=org_id,
            period=period,
            steps_this_period=steps,
            robots_active=robots_active,
            storage_mb=storage,
            plugins_loaded=plugins_loaded,
            plan_limits=_PLAN_LIMITS.get(plan, _PLAN_LIMITS["free"]),
            days_in_period=days,
        )

    def check_quota(self, org_id: str, resource: str,
                    current_used: Optional[int] = None) -> QuotaCheckResult:
        plan = self._get_plan(org_id)
        limits = _PLAN_LIMITS.get(plan, _PLAN_LIMITS["free"])

        if resource == "steps":
            limit = limits["steps_per_month"]
            used  = current_used if current_used is not None else self.steps_in_period(org_id, 30)
        elif resource == "storage":
            limit = int(limits["storage_mb"])
            used  = current_used if current_used is not None else int(self.storage_mb(org_id))
        elif resource == "robots":
            limit = limits["robots"]
            used  = current_used if current_used is not None else self.active_robots(org_id)
        elif resource == "plugins":
            limit = limits["plugins"]
            used  = current_used if current_used is not None else 0
        else:
            limit = 999_999_999
            used  = current_used or 0

        pct = (used / limit * 100) if limit > 0 else 100.0
        if pct >= 100.0:
            status = QuotaStatusEnum.EXCEEDED
        elif pct >= 80.0:
            status = QuotaStatusEnum.WARNING
        else:
            status = QuotaStatusEnum.OK

        return QuotaCheckResult(resource=resource, used=used, limit=limit,
                                status=status, pct=pct)

    def all_quotas(self, org_id: str, robots_active: int = 0,
                   plugins_loaded: int = 0) -> Dict[str, QuotaCheckResult]:
        return {
            "steps":   self.check_quota(org_id, "steps"),
            "storage": self.check_quota(org_id, "storage"),
            "robots":  self.check_quota(org_id, "robots", robots_active),
            "plugins": self.check_quota(org_id, "plugins", plugins_loaded),
        }

    def close(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()


def _day_cutoff(days: int) -> str:
    import datetime
    d = datetime.date.today() - datetime.timedelta(days=days)
    return d.isoformat()


def get_metering(db_path: Optional[Path] = None) -> UsageMetering:
    global _metering_instance
    with _metering_lock:
        if _metering_instance is None:
            _metering_instance = UsageMetering(db_path)
    return _metering_instance
