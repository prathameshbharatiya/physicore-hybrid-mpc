"""physicore/api/audit.py — Immutable audit log for all mutating API actions."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH: Path = Path.home() / ".physicore" / "audit.db"
_log_instance: Optional["AuditLog"] = None
_log_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# AuditEvent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditEvent:
    event_id: str
    user_id: str
    org_id: str
    action: str      # e.g. "engine.configure", "plugin.install", "org.invite"
    resource: str    # e.g. "engine", "plugin:my_plugin", "org:abc123"
    details: Dict[str, Any]
    ip: str
    timestamp: float
    status: str = "ok"   # "ok" | "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "ip": self.ip,
            "timestamp": self.timestamp,
            "status": self.status,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "AuditEvent":
        return cls(
            event_id=row["event_id"],
            user_id=row.get("user_id", ""),
            org_id=row.get("org_id", ""),
            action=row["action"],
            resource=row.get("resource", ""),
            details=json.loads(row["details"]) if isinstance(row.get("details"), str) else row.get("details", {}),
            ip=row.get("ip", ""),
            timestamp=row["timestamp"],
            status=row.get("status", "ok"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# AuditLog
# ─────────────────────────────────────────────────────────────────────────────

class AuditLog:
    """SQLite-backed append-only audit log."""

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
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id  TEXT PRIMARY KEY,
                    user_id   TEXT DEFAULT '',
                    org_id    TEXT DEFAULT '',
                    action    TEXT NOT NULL,
                    resource  TEXT DEFAULT '',
                    details   TEXT DEFAULT '{}',
                    ip        TEXT DEFAULT '',
                    timestamp REAL NOT NULL,
                    status    TEXT DEFAULT 'ok'
                );
                CREATE INDEX IF NOT EXISTS idx_audit_org   ON audit_events(org_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_user  ON audit_events(user_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_events(action);
            """)

    # ── Write ─────────────────────────────────────────────────────────────────

    def log(self, user_id: str, org_id: str, action: str, resource: str = "",
            details: Optional[Dict[str, Any]] = None, ip: str = "",
            status: str = "ok") -> AuditEvent:
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            org_id=org_id,
            action=action,
            resource=resource,
            details=details or {},
            ip=ip,
            timestamp=time.time(),
            status=status,
        )
        with self._lock, self._conn() as con:
            con.execute(
                "INSERT INTO audit_events VALUES (?,?,?,?,?,?,?,?,?)",
                (event.event_id, event.user_id, event.org_id, event.action,
                 event.resource, json.dumps(event.details), event.ip,
                 event.timestamp, event.status),
            )
        return event

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, org_id: str, start_time: float = 0.0, end_time: Optional[float] = None,
              user_id: Optional[str] = None, action: Optional[str] = None,
              limit: int = 100, offset: int = 0) -> List[AuditEvent]:
        if end_time is None:
            end_time = time.time() + 1.0

        where = ["org_id=?", "timestamp>=?", "timestamp<=?"]
        params: List[Any] = [org_id, start_time, end_time]

        if user_id:
            where.append("user_id=?")
            params.append(user_id)
        if action:
            where.append("action LIKE ?")
            params.append(f"%{action}%")

        sql = (
            f"SELECT * FROM audit_events WHERE {' AND '.join(where)} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        )
        params += [limit, offset]

        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [AuditEvent.from_row(dict(r)) for r in rows]

    def query_global(self, start_time: float = 0.0, end_time: Optional[float] = None,
                     action: Optional[str] = None, limit: int = 200) -> List[AuditEvent]:
        if end_time is None:
            end_time = time.time() + 1.0
        where = ["timestamp>=?", "timestamp<=?"]
        params: List[Any] = [start_time, end_time]
        if action:
            where.append("action LIKE ?")
            params.append(f"%{action}%")
        sql = (
            f"SELECT * FROM audit_events WHERE {' AND '.join(where)} "
            f"ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [AuditEvent.from_row(dict(r)) for r in rows]

    def count(self, org_id: str) -> int:
        with self._conn() as con:
            return con.execute(
                "SELECT COUNT(*) FROM audit_events WHERE org_id=?", (org_id,)
            ).fetchone()[0]

    # ── Export ────────────────────────────────────────────────────────────────

    def export_csv(self, org_id: str, period_days: int = 30) -> str:
        start = time.time() - period_days * 86400
        events = self.query(org_id, start_time=start, limit=10000)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=[
            "event_id", "user_id", "org_id", "action", "resource",
            "ip", "timestamp", "status",
        ])
        writer.writeheader()
        for ev in events:
            row = ev.to_dict()
            row.pop("details", None)
            writer.writerow(row)
        return buf.getvalue()

    def close(self) -> None:
        pass


def get_audit_log(db_path: Optional[Path] = None) -> AuditLog:
    global _log_instance
    with _log_lock:
        if _log_instance is None:
            _log_instance = AuditLog(db_path)
    return _log_instance
