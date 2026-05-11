"""
physicore/data/telemetry_store.py — SQLite-backed time-series telemetry store

Lightweight embedded time-series database (no external services required).
All writes are non-blocking: they go onto a queue that a background thread drains.
WAL mode is enabled so reads never block writes.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_DB = Path.home() / ".physicore" / "telemetry.db"


# ── Session record ─────────────────────────────────────────────────────────────

@dataclass
class SessionRecord:
    session_id:  str
    robot_id:    str
    platform:    str
    started_at:  float
    ended_at:    float = 0.0
    step_count:  int   = 0
    meta:        Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── TelemetryStore ────────────────────────────────────────────────────────────

class TelemetryStore:
    """
    Embedded time-series database backed by SQLite.

    Schema
    ------
    telemetry(robot_id, session_id, step, timestamp, key, value)
    sessions(session_id PK, robot_id, platform, started_at, ended_at, step_count, meta)
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB):
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._queue:  queue.Queue = queue.Queue(maxsize=50_000)
        self._stop    = threading.Event()
        self._lock    = threading.Lock()

        # Initialise schema on the *writer thread's* connection later;
        # open a separate read connection for the calling thread.
        self._init_db(self._db_path)

        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="telemetry-writer"
        )
        self._writer_thread.start()

    # ── Schema init ───────────────────────────────────────────────────────────

    def _init_db(self, db_path: Path) -> None:
        con = sqlite3.connect(str(db_path), check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS telemetry (
                robot_id   TEXT NOT NULL,
                session_id TEXT NOT NULL,
                step       INTEGER NOT NULL,
                timestamp  REAL NOT NULL,
                key        TEXT NOT NULL,
                value      REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tel_sid  ON telemetry(session_id);
            CREATE INDEX IF NOT EXISTS idx_tel_rid  ON telemetry(robot_id, key, timestamp);

            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                robot_id    TEXT NOT NULL,
                platform    TEXT NOT NULL,
                started_at  REAL NOT NULL,
                ended_at    REAL NOT NULL DEFAULT 0,
                step_count  INTEGER NOT NULL DEFAULT 0,
                meta        TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_ses_rid ON sessions(robot_id);
        """)
        con.commit()
        con.close()

    # ── Writer loop ───────────────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        con = sqlite3.connect(str(self._db_path), check_same_thread=True)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")

        BATCH = 256
        pending: list = []
        last_flush = time.monotonic()

        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.05)
                pending.append(item)
            except queue.Empty:
                pass

            now = time.monotonic()
            if len(pending) >= BATCH or (pending and now - last_flush > 0.1):
                self._flush(con, pending)
                pending.clear()
                last_flush = now

        if pending:
            self._flush(con, pending)
        con.close()

    def _flush(self, con: sqlite3.Connection, items: list) -> None:
        rows = [r for r in items if r[0] == "row"]
        sess = [r for r in items if r[0] == "session"]
        upds = [r for r in items if r[0] == "update"]

        if rows:
            con.executemany(
                "INSERT INTO telemetry(robot_id,session_id,step,timestamp,key,value)"
                " VALUES(?,?,?,?,?,?)",
                [r[1] for r in rows]
            )
        for _, sid, sc, ea in upds:
            con.execute(
                "UPDATE sessions SET step_count=?, ended_at=? WHERE session_id=?",
                (sc, ea, sid)
            )
        for _, rec in sess:
            con.execute(
                "INSERT OR REPLACE INTO sessions"
                "(session_id,robot_id,platform,started_at,ended_at,step_count,meta)"
                " VALUES(?,?,?,?,?,?,?)",
                (rec.session_id, rec.robot_id, rec.platform,
                 rec.started_at, rec.ended_at, rec.step_count,
                 json.dumps(rec.meta))
            )
        con.commit()

    # ── Public API ────────────────────────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        robot_id:   str,
        platform:   str,
        meta:       Optional[dict] = None,
    ) -> SessionRecord:
        rec = SessionRecord(
            session_id = session_id,
            robot_id   = robot_id,
            platform   = platform,
            started_at = time.time(),
            meta       = meta or {},
        )
        self._queue.put(("session", rec))
        return rec

    def write(
        self,
        robot_id:   str,
        session_id: str,
        step:       int,
        timestamp:  float,
        metrics:    Dict[str, float],
    ) -> None:
        """Non-blocking: enqueues all metrics for background insertion."""
        for key, value in metrics.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            self._queue.put_nowait(
                ("row", (robot_id, session_id, step, timestamp, key, v))
            )

    def flush_session(self, session_id: str, step_count: int) -> None:
        """Update the sessions table with final step count and ended_at."""
        self._queue.put(("update", session_id, step_count, time.time()))

    # ── Query API ─────────────────────────────────────────────────────────────

    def _read_con(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path), check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.row_factory = sqlite3.Row
        return con

    def query(
        self,
        robot_id:   str,
        key:        str,
        start_time: Optional[float] = None,
        end_time:   Optional[float] = None,
        limit:      int = 1000,
    ) -> List[Dict[str, float]]:
        """Return [{timestamp, value}] for a specific metric."""
        con  = self._read_con()
        args: list = [robot_id, key]
        sql  = "SELECT timestamp, value FROM telemetry WHERE robot_id=? AND key=?"
        if start_time is not None:
            sql += " AND timestamp>=?"; args.append(start_time)
        if end_time is not None:
            sql += " AND timestamp<=?"; args.append(end_time)
        sql += " ORDER BY timestamp ASC LIMIT ?"
        args.append(limit)
        rows = con.execute(sql, args).fetchall()
        con.close()
        return [{"timestamp": r["timestamp"], "value": r["value"]} for r in rows]

    def query_session(
        self,
        session_id: str,
        keys: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, float]]]:
        """Return DataFrame-compatible dict: {key: [{step, timestamp, value}]}"""
        con  = self._read_con()
        args: list = [session_id]
        sql  = "SELECT step, timestamp, key, value FROM telemetry WHERE session_id=?"
        if keys:
            placeholders = ",".join("?" * len(keys))
            sql += f" AND key IN ({placeholders})"
            args.extend(keys)
        sql += " ORDER BY step ASC"
        rows = con.execute(sql, args).fetchall()
        con.close()

        result: Dict[str, List] = {}
        for r in rows:
            k = r["key"]
            if k not in result:
                result[k] = []
            result[k].append({
                "step":      r["step"],
                "timestamp": r["timestamp"],
                "value":     r["value"],
            })
        return result

    def sessions(
        self,
        robot_id: Optional[str] = None,
        limit:    int = 50,
    ) -> List[SessionRecord]:
        con  = self._read_con()
        if robot_id:
            sql  = ("SELECT * FROM sessions WHERE robot_id=?"
                    " ORDER BY started_at DESC LIMIT ?")
            rows = con.execute(sql, [robot_id, limit]).fetchall()
        else:
            sql  = "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?"
            rows = con.execute(sql, [limit]).fetchall()
        con.close()
        return [
            SessionRecord(
                session_id = r["session_id"],
                robot_id   = r["robot_id"],
                platform   = r["platform"],
                started_at = r["started_at"],
                ended_at   = r["ended_at"],
                step_count = r["step_count"],
                meta       = json.loads(r["meta"] or "{}"),
            )
            for r in rows
        ]

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        con  = self._read_con()
        row  = con.execute(
            "SELECT * FROM sessions WHERE session_id=?", [session_id]
        ).fetchone()
        con.close()
        if row is None:
            return None
        return SessionRecord(
            session_id = row["session_id"],
            robot_id   = row["robot_id"],
            platform   = row["platform"],
            started_at = row["started_at"],
            ended_at   = row["ended_at"],
            step_count = row["step_count"],
            meta       = json.loads(row["meta"] or "{}"),
        )

    def export_csv(self, session_id: str, output_path: str | Path) -> None:
        """Export all metrics for a session to CSV."""
        con  = self._read_con()
        rows = con.execute(
            "SELECT robot_id, session_id, step, timestamp, key, value"
            " FROM telemetry WHERE session_id=? ORDER BY step, key",
            [session_id]
        ).fetchall()
        con.close()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["robot_id", "session_id", "step", "timestamp", "key", "value"])
            for r in rows:
                writer.writerow([r["robot_id"], r["session_id"],
                                 r["step"], r["timestamp"], r["key"], r["value"]])

    def export_json(self, session_id: str) -> dict:
        """Export full session (metadata + time series) as a JSON-serialisable dict."""
        session = self.get_session(session_id)
        data    = self.query_session(session_id)
        return {
            "session":  session.to_dict() if session else {},
            "series":   data,
        }

    def purge_old(self, days: int = 30) -> int:
        """Delete sessions and their telemetry older than `days` days. Returns deleted count."""
        cutoff = time.time() - days * 86400
        con    = sqlite3.connect(str(self._db_path), check_same_thread=False)
        cur    = con.execute(
            "SELECT session_id FROM sessions WHERE started_at < ?", [cutoff]
        )
        old_ids = [r[0] for r in cur.fetchall()]
        if old_ids:
            placeholders = ",".join("?" * len(old_ids))
            con.execute(f"DELETE FROM telemetry WHERE session_id IN ({placeholders})",
                        old_ids)
            con.execute(f"DELETE FROM sessions WHERE session_id IN ({placeholders})",
                        old_ids)
        con.commit()
        con.close()
        return len(old_ids)

    def delete_session(self, session_id: str) -> bool:
        con = sqlite3.connect(str(self._db_path), check_same_thread=False)
        cur = con.execute(
            "DELETE FROM sessions WHERE session_id=?", [session_id]
        )
        con.execute("DELETE FROM telemetry WHERE session_id=?", [session_id])
        con.commit()
        deleted = cur.rowcount > 0
        con.close()
        return deleted

    def stats(self) -> dict:
        """Return aggregate statistics about the database."""
        con            = self._read_con()
        total_sessions = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        total_rows     = con.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
        con.close()
        size_bytes     = self._db_path.stat().st_size if self._db_path.exists() else 0
        return {
            "total_sessions": total_sessions,
            "total_rows":     total_rows,
            "db_size_mb":     round(size_bytes / (1024 * 1024), 4),
            "db_path":        str(self._db_path),
            "queue_depth":    self._queue.qsize(),
        }

    def close(self) -> None:
        self._stop.set()
        self._writer_thread.join(timeout=5)


# ── Singleton ─────────────────────────────────────────────────────────────────

_store: Optional[TelemetryStore] = None


def get_store(db_path: str | Path = _DEFAULT_DB) -> TelemetryStore:
    global _store
    if _store is None:
        _store = TelemetryStore(db_path)
    return _store
