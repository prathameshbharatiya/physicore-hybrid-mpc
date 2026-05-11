"""physicore/sdk/marketplace.py — Plugin marketplace registry and safety scanner."""

from __future__ import annotations

import ast
import io
import json
import re
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from physicore.sdk.plugin_manifest import PluginManifest, validate_manifest

MARKETPLACE_CATEGORIES = [
    "perception", "safety", "analytics", "actuation", "communication", "demo"
]

_STORE_ROOT: Path = Path.home() / ".physicore" / "marketplace"
_registry_instance: Optional["MarketplaceRegistry"] = None
_registry_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Review:
    review_id: str
    author_id: str
    rating: float          # 1–5
    text: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "author_id": self.author_id,
            "rating": self.rating,
            "text": self.text,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Review":
        return cls(**d)


@dataclass
class MarketplaceEntry:
    manifest: PluginManifest
    author_id: str
    category: str
    download_count: int = 0
    rating: float = 0.0
    reviews: List[Review] = field(default_factory=list)
    verified: bool = False
    price: float = 0.0
    screenshots: List[str] = field(default_factory=list)
    submitted_at: float = field(default_factory=time.time)
    versions: List[str] = field(default_factory=list)
    zip_path: Optional[str] = None   # path to stored zip

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id":      self.manifest.plugin_id,
            "name":           self.manifest.name,
            "version":        self.manifest.version,
            "description":    self.manifest.description,
            "author":         self.manifest.author,
            "author_id":      self.author_id,
            "category":       self.category,
            "tags":           self.manifest.tags,
            "download_count": self.download_count,
            "rating":         round(self.rating, 2),
            "reviews":        [r.to_dict() for r in self.reviews],
            "verified":       self.verified,
            "price":          self.price,
            "screenshots":    self.screenshots,
            "submitted_at":   self.submitted_at,
            "versions":       self.versions,
            "manifest":       self.manifest.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarketplaceEntry":
        manifest = PluginManifest.from_dict(d["manifest"])
        return cls(
            manifest=manifest,
            author_id=d.get("author_id", ""),
            category=d.get("category", "demo"),
            download_count=d.get("download_count", 0),
            rating=d.get("rating", 0.0),
            reviews=[Review.from_dict(r) for r in d.get("reviews", [])],
            verified=d.get("verified", False),
            price=d.get("price", 0.0),
            screenshots=d.get("screenshots", []),
            submitted_at=d.get("submitted_at", time.time()),
            versions=d.get("versions", [manifest.version]),
            zip_path=d.get("zip_path"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Safety scanner
# ─────────────────────────────────────────────────────────────────────────────

# Forbidden patterns in plugin source code
_FORBIDDEN_PATTERNS = [
    (re.compile(r"\bsocket\s*\("), "raw socket usage"),
    (re.compile(r"\bsubprocess\s*\."), "subprocess execution"),
    (re.compile(r"\bos\.system\s*\("), "os.system call"),
    (re.compile(r"\beval\s*\("), "eval() call"),
    (re.compile(r"\bexec\s*\("), "exec() call"),
    (re.compile(r"\bopen\s*\([^)]*['\"][^'\"]+['\"]"), "file open outside sandbox"),
    (re.compile(r"\b__import__\s*\("), "__import__ call"),
    (re.compile(r"\burllib\b.*\bopen\b"), "network call via urllib"),
    (re.compile(r"\brequests\s*\."), "network call via requests"),
    (re.compile(r"\bhttpx\s*\."), "network call via httpx"),
    (re.compile(r"\baiohttp\s*\."), "network call via aiohttp"),
]

_ALLOWED_OPEN_RE = re.compile(r"open\s*\(\s*['\"]?/plugins/")


class SafetyScanResult:
    def __init__(self):
        self.passed = True
        self.violations: List[str] = []

    def fail(self, reason: str) -> None:
        self.passed = False
        self.violations.append(reason)

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "violations": self.violations}


def scan_source(source_code: str, filename: str = "<plugin>") -> SafetyScanResult:
    """
    Static analysis safety scan of plugin Python source code.
    Rejects: raw network calls, subprocess, eval/exec, file access outside /plugins/.
    """
    result = SafetyScanResult()
    lines = source_code.splitlines()

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for pattern, reason in _FORBIDDEN_PATTERNS:
            if pattern.search(line):
                # Exception: file open to /plugins/ path
                if "open" in reason and _ALLOWED_OPEN_RE.search(line):
                    continue
                result.fail(f"{filename}:{lineno}: {reason} — {line.strip()[:80]}")

    # AST-level checks
    try:
        tree = ast.parse(source_code, filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("socket", "subprocess", "ctypes", "cffi"):
                        result.fail(f"Forbidden import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in ("socket", "subprocess", "ctypes"):
                    result.fail(f"Forbidden from-import: {node.module}")
    except SyntaxError as exc:
        result.fail(f"Syntax error: {exc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MarketplaceRegistry
# ─────────────────────────────────────────────────────────────────────────────

class MarketplaceRegistry:
    """
    File-backed plugin marketplace.  Plugins are stored as zip archives under
    STORE_ROOT/plugins/{plugin_id}/{version}.physicore-plugin
    with a catalog.json index.
    """

    def __init__(self, store_root: Optional[Path] = None):
        self._root = Path(store_root) if store_root else _STORE_ROOT
        self._plugins_dir = self._root / "plugins"
        self._catalog_path = self._root / "catalog.json"
        self._lock = threading.Lock()
        self._root.mkdir(parents=True, exist_ok=True)
        self._plugins_dir.mkdir(exist_ok=True)
        self._catalog: Dict[str, Dict[str, Any]] = self._load_catalog()

    # ── Catalog I/O ───────────────────────────────────────────────────────────

    def _load_catalog(self) -> Dict[str, Dict[str, Any]]:
        if self._catalog_path.exists():
            try:
                return json.loads(self._catalog_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_catalog(self) -> None:
        self._catalog_path.write_text(
            json.dumps(self._catalog, indent=2), encoding="utf-8"
        )

    # ── Submit ────────────────────────────────────────────────────────────────

    def submit(self, plugin_zip: bytes, author_id: str,
               category: str = "demo", price: float = 0.0,
               screenshots: Optional[List[str]] = None) -> MarketplaceEntry:
        """
        Validate manifest + safety scan, then store the plugin zip.
        Raises ValueError on invalid manifest or safety violations.
        """
        # Parse zip and extract manifest + source files
        try:
            zf_buf = io.BytesIO(plugin_zip)
            with zipfile.ZipFile(zf_buf, "r") as zf:
                names = zf.namelist()
                # Find manifest
                json_files = [n for n in names if n.endswith("plugin.json")]
                if not json_files:
                    raise ValueError("Missing plugin.json in zip")
                raw_json = zf.read(json_files[0]).decode("utf-8")
                manifest_data = json.loads(raw_json)
                manifest = validate_manifest(manifest_data)

                # Safety scan all .py files
                py_files = [n for n in names if n.endswith(".py")]
                for py_name in py_files:
                    src = zf.read(py_name).decode("utf-8", errors="replace")
                    scan = scan_source(src, filename=py_name)
                    if not scan.passed:
                        raise ValueError(
                            f"Safety scan failed for {py_name}: "
                            + "; ".join(scan.violations[:3])
                        )
        except (zipfile.BadZipFile, KeyError) as exc:
            raise ValueError(f"Invalid plugin zip: {exc}") from exc

        if category not in MARKETPLACE_CATEGORIES:
            category = "demo"

        # Store zip
        plugin_dir = self._plugins_dir / manifest.plugin_id
        plugin_dir.mkdir(exist_ok=True)
        zip_path = plugin_dir / f"{manifest.version}.physicore-plugin"
        zip_path.write_bytes(plugin_zip)

        with self._lock:
            existing = self._catalog.get(manifest.plugin_id)
            if existing:
                entry = MarketplaceEntry.from_dict(existing)
                if manifest.version not in entry.versions:
                    entry.versions.append(manifest.version)
                entry.manifest = manifest
            else:
                entry = MarketplaceEntry(
                    manifest=manifest,
                    author_id=author_id,
                    category=category,
                    price=price,
                    screenshots=screenshots or [],
                    versions=[manifest.version],
                    zip_path=str(zip_path),
                )
            entry.zip_path = str(zip_path)
            self._catalog[manifest.plugin_id] = entry.to_dict()
            self._save_catalog()

        return entry

    # ── Get / Search ──────────────────────────────────────────────────────────

    def get(self, plugin_id: str) -> Optional[MarketplaceEntry]:
        d = self._catalog.get(plugin_id)
        return MarketplaceEntry.from_dict(d) if d else None

    def search(self, query: str = "", platform: Optional[str] = None,
               category: Optional[str] = None, free_only: bool = False,
               verified_only: bool = False, limit: int = 50) -> List[MarketplaceEntry]:
        results: List[MarketplaceEntry] = []
        q = query.lower()
        for d in self._catalog.values():
            entry = MarketplaceEntry.from_dict(d)
            if q and not any(
                q in (entry.manifest.name or "").lower()
                or q in (entry.manifest.description or "").lower()
                or q in " ".join(entry.manifest.tags).lower()
                for _ in [1]   # single iteration
            ):
                # Check each field individually
                name_match = q in entry.manifest.name.lower()
                desc_match = q in entry.manifest.description.lower()
                tag_match  = any(q in t.lower() for t in entry.manifest.tags)
                cat_match  = q in entry.category.lower()
                if not (name_match or desc_match or tag_match or cat_match):
                    continue
            if category and entry.category != category:
                continue
            if free_only and entry.price > 0:
                continue
            if verified_only and not entry.verified:
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        # Sort by download count descending
        results.sort(key=lambda e: e.download_count, reverse=True)
        return results

    def list_all(self, limit: int = 100) -> List[MarketplaceEntry]:
        return [MarketplaceEntry.from_dict(d) for d in list(self._catalog.values())[:limit]]

    # ── Install ───────────────────────────────────────────────────────────────

    def install(self, plugin_id: str, version: str, target_dir: str) -> Path:
        """
        Copy the plugin zip to target_dir and return the installed path.
        Raises FileNotFoundError if plugin/version not found.
        """
        entry = self.get(plugin_id)
        if entry is None:
            raise FileNotFoundError(f"Plugin {plugin_id!r} not in marketplace")

        zip_path = self._plugins_dir / plugin_id / f"{version}.physicore-plugin"
        if not zip_path.exists():
            raise FileNotFoundError(f"Version {version!r} of {plugin_id!r} not found")

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        dest = target / zip_path.name
        import shutil
        shutil.copy2(zip_path, dest)

        # Increment download count
        with self._lock:
            d = self._catalog.get(plugin_id, {})
            d["download_count"] = d.get("download_count", 0) + 1
            self._catalog[plugin_id] = d
            self._save_catalog()

        return dest

    # ── Rate ──────────────────────────────────────────────────────────────────

    def rate(self, plugin_id: str, author_id: str, rating: float, text: str = "") -> Review:
        if not (1.0 <= rating <= 5.0):
            raise ValueError("Rating must be between 1 and 5")
        entry = self.get(plugin_id)
        if entry is None:
            raise FileNotFoundError(f"Plugin {plugin_id!r} not found")

        review = Review(
            review_id=str(uuid.uuid4()),
            author_id=author_id,
            rating=float(rating),
            text=text,
            created_at=time.time(),
        )
        with self._lock:
            d = self._catalog[plugin_id]
            reviews = d.get("reviews", [])
            # Replace existing review from same author
            reviews = [r for r in reviews if r.get("author_id") != author_id]
            reviews.append(review.to_dict())
            d["reviews"] = reviews
            # Recalculate average
            d["rating"] = sum(r["rating"] for r in reviews) / len(reviews) if reviews else 0.0
            self._catalog[plugin_id] = d
            self._save_catalog()

        return review

    def mark_verified(self, plugin_id: str) -> bool:
        with self._lock:
            d = self._catalog.get(plugin_id)
            if d is None:
                return False
            d["verified"] = True
            self._catalog[plugin_id] = d
            self._save_catalog()
        return True


def get_marketplace(store_root: Optional[Path] = None) -> MarketplaceRegistry:
    global _registry_instance
    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = MarketplaceRegistry(store_root)
    return _registry_instance
