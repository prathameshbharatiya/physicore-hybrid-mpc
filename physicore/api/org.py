"""physicore/api/org.py — Organizations, teams, membership, and quota management."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH: Path = Path.home() / ".physicore" / "orgs.db"
_store_instance: Optional["OrgStore"] = None
_store_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class OrgPlan(str, Enum):
    FREE       = "free"
    PRO        = "pro"
    ENTERPRISE = "enterprise"

    def robot_quota(self) -> int:
        return {"free": 3, "pro": 20, "enterprise": 500}[self.value]

    def plugin_quota(self) -> int:
        return {"free": 5, "pro": 50, "enterprise": 500}[self.value]

    def retention_days(self) -> int:
        return {"free": 7, "pro": 90, "enterprise": 365}[self.value]


class OrgRole(str, Enum):
    OWNER  = "owner"
    ADMIN  = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

    def permissions(self) -> List[str]:
        base = {
            "owner":  ["read", "write", "admin", "billing", "invite", "remove"],
            "admin":  ["read", "write", "invite", "remove"],
            "member": ["read", "write"],
            "viewer": ["read"],
        }
        return base[self.value]


@dataclass
class Organization:
    org_id: str
    name: str
    plan: str = OrgPlan.FREE.value
    created_at: float = field(default_factory=time.time)
    member_ids: List[str] = field(default_factory=list)
    robot_quota: int = 3
    plugin_quota: int = 5
    data_retention_days: int = 7
    owner_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "org_id": self.org_id,
            "name": self.name,
            "plan": self.plan,
            "created_at": self.created_at,
            "member_ids": self.member_ids,
            "robot_quota": self.robot_quota,
            "plugin_quota": self.plugin_quota,
            "data_retention_days": self.data_retention_days,
            "owner_id": self.owner_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Organization":
        return cls(
            org_id=d["org_id"],
            name=d["name"],
            plan=d.get("plan", OrgPlan.FREE.value),
            created_at=d.get("created_at", time.time()),
            member_ids=json.loads(d["member_ids"]) if isinstance(d.get("member_ids"), str) else d.get("member_ids", []),
            robot_quota=d.get("robot_quota", 3),
            plugin_quota=d.get("plugin_quota", 5),
            data_retention_days=d.get("data_retention_days", 7),
            owner_id=d.get("owner_id", ""),
        )


@dataclass
class OrgMembership:
    user_id: str
    org_id: str
    role: str = OrgRole.MEMBER.value
    joined_at: float = field(default_factory=time.time)
    email: str = ""
    permissions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.permissions:
            self.permissions = OrgRole(self.role).permissions()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "org_id": self.org_id,
            "role": self.role,
            "joined_at": self.joined_at,
            "email": self.email,
            "permissions": self.permissions,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OrgMembership":
        return cls(
            user_id=d["user_id"],
            org_id=d["org_id"],
            role=d.get("role", OrgRole.MEMBER.value),
            joined_at=d.get("joined_at", time.time()),
            email=d.get("email", ""),
            permissions=json.loads(d["permissions"]) if isinstance(d.get("permissions"), str) else d.get("permissions", []),
        )


@dataclass
class QuotaStatus:
    resource: str
    used: int
    limit: int
    pct: float
    status: str  # "ok" | "warning" | "exceeded"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource,
            "used": self.used,
            "limit": self.limit,
            "pct": round(self.pct, 1),
            "status": self.status,
        }


# ─────────────────────────────────────────────────────────────────────────────
# OrgStore
# ─────────────────────────────────────────────────────────────────────────────

class OrgStore:
    """SQLite-backed organization and membership store."""

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
                CREATE TABLE IF NOT EXISTS orgs (
                    org_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    plan TEXT DEFAULT 'free',
                    created_at REAL,
                    member_ids TEXT DEFAULT '[]',
                    robot_quota INTEGER DEFAULT 3,
                    plugin_quota INTEGER DEFAULT 5,
                    data_retention_days INTEGER DEFAULT 7,
                    owner_id TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS memberships (
                    user_id TEXT NOT NULL,
                    org_id TEXT NOT NULL,
                    role TEXT DEFAULT 'member',
                    joined_at REAL,
                    email TEXT DEFAULT '',
                    permissions TEXT DEFAULT '[]',
                    PRIMARY KEY (user_id, org_id)
                );
                CREATE TABLE IF NOT EXISTS invites (
                    invite_id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    email TEXT NOT NULL,
                    role TEXT DEFAULT 'member',
                    invited_at REAL,
                    accepted INTEGER DEFAULT 0
                );
            """)

    # ── Org CRUD ──────────────────────────────────────────────────────────────

    def create_org(self, name: str, owner_id: str, plan: str = OrgPlan.FREE.value) -> Organization:
        org_plan = OrgPlan(plan)
        org = Organization(
            org_id=str(uuid.uuid4()),
            name=name,
            plan=plan,
            created_at=time.time(),
            member_ids=[owner_id],
            robot_quota=org_plan.robot_quota(),
            plugin_quota=org_plan.plugin_quota(),
            data_retention_days=org_plan.retention_days(),
            owner_id=owner_id,
        )
        with self._lock, self._conn() as con:
            con.execute(
                "INSERT INTO orgs VALUES (?,?,?,?,?,?,?,?,?)",
                (org.org_id, org.name, org.plan, org.created_at,
                 json.dumps(org.member_ids), org.robot_quota,
                 org.plugin_quota, org.data_retention_days, org.owner_id),
            )
            con.execute(
                "INSERT INTO memberships VALUES (?,?,?,?,?,?)",
                (owner_id, org.org_id, OrgRole.OWNER.value, org.created_at,
                 "", json.dumps(OrgRole.OWNER.permissions())),
            )
        return org

    def get_org(self, org_id: str) -> Optional[Organization]:
        with self._conn() as con:
            row = con.execute("SELECT * FROM orgs WHERE org_id=?", (org_id,)).fetchone()
        if row is None:
            return None
        return Organization.from_dict(dict(row))

    def update_org(self, org_id: str, **kwargs) -> Optional[Organization]:
        allowed = {"name", "plan", "robot_quota", "plugin_quota", "data_retention_days"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_org(org_id)
        sets = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [org_id]
        with self._lock, self._conn() as con:
            con.execute(f"UPDATE orgs SET {sets} WHERE org_id=?", vals)
        return self.get_org(org_id)

    def delete_org(self, org_id: str) -> bool:
        with self._lock, self._conn() as con:
            cur = con.execute("DELETE FROM orgs WHERE org_id=?", (org_id,))
            con.execute("DELETE FROM memberships WHERE org_id=?", (org_id,))
        return cur.rowcount > 0

    def list_orgs_for_user(self, user_id: str) -> List[Organization]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT org_id FROM memberships WHERE user_id=?", (user_id,)
            ).fetchall()
        orgs = []
        for row in rows:
            org = self.get_org(row["org_id"])
            if org:
                orgs.append(org)
        return orgs

    # ── Membership ────────────────────────────────────────────────────────────

    def get_members(self, org_id: str) -> List[OrgMembership]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM memberships WHERE org_id=?", (org_id,)
            ).fetchall()
        return [OrgMembership.from_dict(dict(r)) for r in rows]

    def get_membership(self, user_id: str, org_id: str) -> Optional[OrgMembership]:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM memberships WHERE user_id=? AND org_id=?",
                (user_id, org_id),
            ).fetchone()
        return OrgMembership.from_dict(dict(row)) if row else None

    def add_member(self, org_id: str, user_id: str, role: str = OrgRole.MEMBER.value,
                   email: str = "") -> OrgMembership:
        perms = OrgRole(role).permissions()
        mem = OrgMembership(
            user_id=user_id, org_id=org_id, role=role,
            joined_at=time.time(), email=email, permissions=perms,
        )
        with self._lock, self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO memberships VALUES (?,?,?,?,?,?)",
                (user_id, org_id, role, mem.joined_at, email, json.dumps(perms)),
            )
            # update member_ids in org
            org = self.get_org(org_id)
            if org and user_id not in org.member_ids:
                org.member_ids.append(user_id)
                con.execute(
                    "UPDATE orgs SET member_ids=? WHERE org_id=?",
                    (json.dumps(org.member_ids), org_id),
                )
        return mem

    def change_role(self, org_id: str, user_id: str, new_role: str) -> Optional[OrgMembership]:
        perms = OrgRole(new_role).permissions()
        with self._lock, self._conn() as con:
            cur = con.execute(
                "UPDATE memberships SET role=?, permissions=? WHERE user_id=? AND org_id=?",
                (new_role, json.dumps(perms), user_id, org_id),
            )
        if cur.rowcount == 0:
            return None
        return self.get_membership(user_id, org_id)

    def remove_member(self, org_id: str, user_id: str) -> bool:
        with self._lock, self._conn() as con:
            cur = con.execute(
                "DELETE FROM memberships WHERE user_id=? AND org_id=?",
                (user_id, org_id),
            )
            org = self.get_org(org_id)
            if org and user_id in org.member_ids:
                org.member_ids.remove(user_id)
                con.execute(
                    "UPDATE orgs SET member_ids=? WHERE org_id=?",
                    (json.dumps(org.member_ids), org_id),
                )
        return cur.rowcount > 0

    # ── Invites ───────────────────────────────────────────────────────────────

    def create_invite(self, org_id: str, email: str, role: str = OrgRole.MEMBER.value) -> str:
        invite_id = str(uuid.uuid4())
        with self._lock, self._conn() as con:
            con.execute(
                "INSERT INTO invites VALUES (?,?,?,?,?,?)",
                (invite_id, org_id, email, role, time.time(), 0),
            )
        return invite_id

    def accept_invite(self, invite_id: str, user_id: str) -> Optional[OrgMembership]:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM invites WHERE invite_id=? AND accepted=0", (invite_id,)
            ).fetchone()
        if row is None:
            return None
        mem = self.add_member(row["org_id"], user_id, row["role"], row["email"])
        with self._lock, self._conn() as con:
            con.execute("UPDATE invites SET accepted=1 WHERE invite_id=?", (invite_id,))
        return mem

    # ── Quota ─────────────────────────────────────────────────────────────────

    def check_quota(self, org_id: str, resource: str, current_used: int) -> QuotaStatus:
        org = self.get_org(org_id)
        if org is None:
            return QuotaStatus(resource=resource, used=current_used, limit=0, pct=100.0, status="exceeded")

        limits = {
            "robots": org.robot_quota,
            "plugins": org.plugin_quota,
        }
        limit = limits.get(resource, 999999)
        pct = (current_used / limit * 100) if limit > 0 else 100.0
        if pct >= 100.0:
            status = "exceeded"
        elif pct >= 80.0:
            status = "warning"
        else:
            status = "ok"
        return QuotaStatus(resource=resource, used=current_used, limit=limit, pct=pct, status=status)

    def get_usage(self, org_id: str, robot_count: int = 0, plugin_count: int = 0,
                  storage_mb: float = 0.0) -> Dict[str, Any]:
        org = self.get_org(org_id)
        if org is None:
            return {}
        return {
            "org_id": org_id,
            "plan": org.plan,
            "robots": self.check_quota(org_id, "robots", robot_count).to_dict(),
            "plugins": self.check_quota(org_id, "plugins", plugin_count).to_dict(),
            "storage_mb": round(storage_mb, 3),
            "retention_days": org.data_retention_days,
        }

    def close(self) -> None:
        pass   # SQLite connections are created per-call


def get_org_store(db_path: Optional[Path] = None) -> OrgStore:
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            _store_instance = OrgStore(db_path)
    return _store_instance
