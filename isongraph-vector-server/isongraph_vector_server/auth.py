"""API keys, tenant records and the registry that authenticates requests.

Key format::

    igv_<tenant-id>.<key-id>.<secret>
         lowercase    16 hex    43 chars urlsafe base64 (256 random bits)

* The server stores only ``HMAC-SHA256(salt, secret)`` plus the salt, never the
  secret. A plain salted fast hash is the right tool here (the same reasoning
  as for GitHub/Stripe style tokens): the secret is 256 random bits, so there
  is nothing for a slow password KDF to protect against, and a per-request KDF
  would cap throughput at a few dozen requests per second per core.
* Verification uses :func:`hmac.compare_digest`. An unknown tenant, unknown
  key id or revoked key still runs one dummy HMAC + compare, so the three
  failure paths take about the same time as a wrong secret.
* The tenant id inside the key is *only a lookup hint* (it lets the stateless
  router pick a shard without a registry). Authority comes from the hash
  check against that tenant's own record: a key naming tenant B but made for
  tenant A fails here.
* A key whose tenant is disabled, or that is revoked, stops working within
  ``recheck_seconds`` (the registry re-stats ``tenant.json`` that often).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import names
from .config import ConfigError, Quotas

KEY_RE = re.compile(r"^igv_([a-z0-9][a-z0-9-]{0,30}[a-z0-9])\.([0-9a-f]{16})\.([A-Za-z0-9_-]{43})$")
TENANT_FORMAT = "igv-tenant"
TENANT_FORMAT_VERSION = 1
ROLES = ("rw", "ro")

_DUMMY_SALT = b"\x00" * 16
_DUMMY_HASH = hmac.new(_DUMMY_SALT, b"dummy", hashlib.sha256).hexdigest()


def _hash_secret(salt: bytes, secret: str) -> str:
    return hmac.new(salt, secret.encode("ascii", "strict"), hashlib.sha256).hexdigest()


def parse_key(token: str) -> Optional[Tuple[str, str, str]]:
    """Split a presented key into (tenant_id, key_id, secret), or None."""
    if not isinstance(token, str) or len(token) > 200:
        return None
    m = KEY_RE.match(token)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def tenant_id_from_key(token: str) -> Optional[str]:
    parsed = parse_key(token)
    return parsed[0] if parsed else None


def new_key_record(role: str = "rw", label: str = "") -> Tuple[Dict[str, Any], str, str]:
    """Return (record_without_tenant, key_id, secret). The caller assembles the token."""
    if role not in ROLES:
        raise ValueError("role must be 'rw' or 'ro'")
    key_id = secrets.token_hex(8)
    secret = secrets.token_urlsafe(32)
    salt = secrets.token_bytes(16)
    record = {
        "key_id": key_id,
        "label": str(label)[:64],
        "role": role,
        "algo": "hmac-sha256",
        "salt": salt.hex(),
        "hash": _hash_secret(salt, secret),
        "created_at": time.time(),
        "revoked_at": None,
    }
    return record, key_id, secret


def make_token(tenant_id: str, key_id: str, secret: str) -> str:
    return f"igv_{tenant_id}.{key_id}.{secret}"


@dataclass(frozen=True)
class Principal:
    """The authenticated caller. Derived only from a verified key."""
    tenant_id: str
    key_id: str
    role: str
    quotas: Quotas


@dataclass
class TenantRecord:
    tenant_id: str
    disabled: bool
    quota_overrides: Dict[str, Any]
    keys: Dict[str, Dict[str, Any]]
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)
    stamp: Tuple[int, int] = (0, 0)


def _atomic_write_json(path: Path, obj: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    data = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def parse_tenant_file(data: Any, expect_id: str) -> TenantRecord:
    """Validate a tenant.json document. Raises ValueError on any problem."""
    if not isinstance(data, dict) or data.get("format") != TENANT_FORMAT:
        raise ValueError("not a tenant file")
    if data.get("version") != TENANT_FORMAT_VERSION:
        raise ValueError("unsupported tenant file version")
    tid = data.get("tenant_id")
    if tid != expect_id or not names.is_tenant_id(tid):
        raise ValueError("tenant id does not match its directory")
    keys: Dict[str, Dict[str, Any]] = {}
    for rec in data.get("keys", []):
        if not isinstance(rec, dict):
            raise ValueError("bad key record")
        kid = rec.get("key_id")
        if not isinstance(kid, str) or not re.fullmatch(r"[0-9a-f]{16}", kid):
            raise ValueError("bad key id")
        if rec.get("role") not in ROLES or rec.get("algo") != "hmac-sha256":
            raise ValueError("bad key role/algo")
        bytes.fromhex(rec["salt"])
        int(rec["hash"], 16)
        keys[kid] = rec
    overrides = data.get("quotas", {})
    if not isinstance(overrides, dict):
        raise ValueError("quotas must be an object")
    return TenantRecord(
        tenant_id=tid,
        disabled=bool(data.get("disabled", False)),
        quota_overrides=overrides,
        keys=keys,
        raw=data,
    )


class TenantRegistry:
    """Read side: authenticate keys against ``<data_dir>/tenants/*/tenant.json``.

    The CLI (another process) is the only writer. The registry notices
    changes by stat'ing each tenant file at most every ``recheck_seconds``
    and re-scans the directory (at most that often) when it sees an unknown
    tenant id, so a freshly created tenant works without a restart while a
    flood of bogus tenant ids cannot turn into a flood of directory scans.
    """

    def __init__(self, data_dir: str, default_quotas: Quotas, recheck_seconds: float = 1.0,
                 clock=time.monotonic):
        self.data_dir = Path(data_dir)
        self.default_quotas = default_quotas
        self.recheck_seconds = recheck_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._records: Dict[str, TenantRecord] = {}
        self._checked_at: Dict[str, float] = {}
        self._last_scan = -1e9
        self.load_errors: Dict[str, str] = {}
        self.rescan(force=True)

    # -- loading ------------------------------------------------------------

    def _tenant_file(self, tid: str) -> Path:
        return names.tenant_dir(self.data_dir, tid) / "tenant.json"

    def _load_one(self, tid: str) -> Optional[TenantRecord]:
        path = self._tenant_file(tid)
        try:
            st = path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            rec = parse_tenant_file(data, tid)
            rec.stamp = stamp
            self.load_errors.pop(tid, None)
            return rec
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Fail closed: a corrupt tenant file authenticates nobody.
            self.load_errors[tid] = f"{type(exc).__name__}"
            return None

    def rescan(self, force: bool = False) -> None:
        now = self._clock()
        with self._lock:
            if not force and now - self._last_scan < self.recheck_seconds:
                return
            self._last_scan = now
            base = self.data_dir / "tenants"
            found = set()
            if base.is_dir():
                for entry in os.scandir(base):
                    if entry.is_dir(follow_symlinks=False) and names.is_tenant_id(entry.name):
                        found.add(entry.name)
            for tid in list(self._records):
                if tid not in found:
                    del self._records[tid]
            for tid in found:
                rec = self._load_one(tid)
                if rec is None:
                    self._records.pop(tid, None)
                else:
                    self._records[tid] = rec
                self._checked_at[tid] = now

    def _fresh(self, tid: str, force: bool = False) -> Optional[TenantRecord]:
        """The current record for ``tid``, re-reading ``tenant.json`` if it changed.

        An unknown tenant costs one targeted file open (not a directory scan);
        a *known* tenant is re-stat'ed at most every ``recheck_seconds`` unless
        ``force`` (used when a presented key id is not in the cached record, so
        a key created a moment ago works immediately). Callers that fail
        authentication are throttled per peer before this is reached.
        """
        now = self._clock()
        with self._lock:
            rec = self._records.get(tid)
            if rec is not None and not force and now - self._checked_at.get(tid, -1e9) < self.recheck_seconds:
                return rec
        try:
            st = self._tenant_file(tid).stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = None
        with self._lock:
            self._checked_at[tid] = now
            cur = self._records.get(tid)
            if cur is not None and stamp is not None and stamp == cur.stamp:
                return cur
            if stamp is None:
                self._records.pop(tid, None)
                return None
        new = self._load_one(tid)
        with self._lock:
            if new is None:
                self._records.pop(tid, None)
            else:
                self._records[tid] = new
            return new

    def tenant_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._records)

    def get(self, tid: str) -> Optional[TenantRecord]:
        return self._fresh(tid) if names.is_tenant_id(tid) else None

    def quotas_for(self, rec: TenantRecord) -> Quotas:
        return self.default_quotas.merged(rec.quota_overrides)

    # -- authentication --------------------------------------------------------

    def authenticate(self, token: str) -> Optional[Principal]:
        """Return the Principal for a valid key, else None. Constant-time compare."""
        parsed = parse_key(token)
        if parsed is None:
            # Still burn one HMAC so malformed tokens are not a timing oracle
            # for "your token had the right shape".
            hmac.compare_digest(_hash_secret(_DUMMY_SALT, "x" * 43), _DUMMY_HASH)
            return None
        tid, kid, secret = parsed
        rec = self._fresh(tid)
        key = rec.keys.get(kid) if rec is not None else None
        if rec is not None and key is None:
            rec = self._fresh(tid, force=True)       # maybe the key was created just now
            key = rec.keys.get(kid) if rec is not None else None
        if key is None or key.get("revoked_at") is not None:
            salt, expected = _DUMMY_SALT, _DUMMY_HASH
            valid_shape = False
        else:
            salt, expected = bytes.fromhex(key["salt"]), key["hash"]
            valid_shape = True
        presented = _hash_secret(salt, secret)
        match = hmac.compare_digest(presented, expected)
        if not (match and valid_shape):
            return None
        assert rec is not None and key is not None
        if rec.disabled:
            return None
        try:
            quotas = self.quotas_for(rec)
        except ConfigError:
            return None   # corrupt quota override: fail closed
        return Principal(tenant_id=tid, key_id=kid, role=key["role"], quotas=quotas)


# -- writer side (used by the CLI) ------------------------------------------------


def tenant_file_path(data_dir: str, tid: str) -> Path:
    return names.tenant_dir(data_dir, tid) / "tenant.json"


def create_tenant(data_dir: str, tid: str, quotas: Optional[Dict[str, Any]] = None) -> Path:
    if not names.is_tenant_id(tid):
        raise ValueError("tenant id must match [a-z0-9][a-z0-9-]{0,30}[a-z0-9]")
    path = tenant_file_path(data_dir, tid)
    if path.exists():
        raise FileExistsError(f"tenant {tid} already exists")
    if quotas:
        Quotas().merged(quotas)   # validates names and values
    doc = {
        "format": TENANT_FORMAT,
        "version": TENANT_FORMAT_VERSION,
        "tenant_id": tid,
        "created_at": time.time(),
        "disabled": False,
        "quotas": quotas or {},
        "keys": [],
    }
    (path.parent / "graphs").mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, doc)
    return path


def _load_raw(data_dir: str, tid: str) -> Tuple[Path, Dict[str, Any]]:
    path = tenant_file_path(data_dir, tid)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    parse_tenant_file(data, tid)
    return path, data


def add_key(data_dir: str, tid: str, role: str = "rw", label: str = "") -> str:
    """Create a key and return the full token. This is the only time it exists."""
    path, data = _load_raw(data_dir, tid)
    record, kid, secret = new_key_record(role, label)
    data["keys"].append(record)
    _atomic_write_json(path, data)
    return make_token(tid, kid, secret)


def revoke_key(data_dir: str, tid: str, key_id: str) -> None:
    path, data = _load_raw(data_dir, tid)
    for rec in data["keys"]:
        if rec["key_id"] == key_id:
            if rec["revoked_at"] is None:
                rec["revoked_at"] = time.time()
            _atomic_write_json(path, data)
            return
    raise KeyError(key_id)


def set_disabled(data_dir: str, tid: str, disabled: bool) -> None:
    path, data = _load_raw(data_dir, tid)
    data["disabled"] = disabled
    _atomic_write_json(path, data)


def set_quotas(data_dir: str, tid: str, quotas: Dict[str, Any]) -> None:
    path, data = _load_raw(data_dir, tid)
    merged = dict(data.get("quotas", {}))
    merged.update(quotas)
    Quotas().merged(merged)
    data["quotas"] = merged
    _atomic_write_json(path, data)


def describe_tenant(data_dir: str, tid: str) -> Dict[str, Any]:
    """Tenant summary for the CLI. Never contains salts or hashes."""
    _, data = _load_raw(data_dir, tid)
    return {
        "tenant_id": tid,
        "disabled": data.get("disabled", False),
        "quotas": data.get("quotas", {}),
        "keys": [{k: r[k] for k in ("key_id", "label", "role", "created_at", "revoked_at")}
                 for r in data["keys"]],
    }
