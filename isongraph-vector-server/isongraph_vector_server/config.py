"""Configuration: a JSON file, overridden by ``IGV_*`` environment variables.

Secrets (the metrics token, the replication token, an embedding endpoint's API
key) are read from the environment only and are never part of the config
file, so a config file can be committed, logged or mounted read-only without
leaking anything. Tenant API keys are not configuration at all: they live,
as salted hashes, in each tenant's ``tenant.json`` (see ``auth.py``).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


class ConfigError(ValueError):
    pass


@dataclass
class Quotas:
    """Per-tenant limits. A tenant's ``tenant.json`` may override any field."""
    max_graphs: int = 20
    max_nodes: int = 100_000          # across all of the tenant's graphs
    max_edges: int = 500_000          # stored edges (an undirected edge counts twice)
    max_dimension: int = 1536         # vector length a graph of this tenant may use
    max_batch_items: int = 500        # nodes/edges per request
    max_top_k: int = 100              # top_k / top_k_results / top_k_seeds cap
    max_hops: int = 5
    max_text_chars: int = 20_000      # one embed text / one string property
    max_properties: int = 64          # properties per node/edge
    rate_per_second: float = 50.0     # sustained requests/s (token bucket refill)
    rate_burst: int = 100             # bucket capacity
    max_inflight: int = 8             # concurrent requests from this tenant

    def merged(self, overrides: Optional[Mapping[str, Any]]) -> "Quotas":
        if not overrides:
            return self
        data = dataclasses.asdict(self)
        for key, value in overrides.items():
            if key not in data:
                raise ConfigError(f"unknown quota {key!r}")
            data[key] = value
        return Quotas(**data).checked()

    def checked(self) -> "Quotas":
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"quota {f.name} must be a number")
            if value <= 0:
                raise ConfigError(f"quota {f.name} must be positive")
        for name in ("max_graphs", "max_nodes", "max_edges", "max_dimension",
                     "max_batch_items", "max_top_k", "max_hops", "max_text_chars",
                     "max_properties", "rate_burst", "max_inflight"):
            if not isinstance(getattr(self, name), int):
                raise ConfigError(f"quota {name} must be an integer")
        return self


@dataclass
class EncoderConfig:
    kind: str = "none"                # none | mock | http | sentence-transformers
    dimension: Optional[int] = None   # mock: required; http: optional (probed)
    model: Optional[str] = None
    url: Optional[str] = None         # http
    api: str = "openai"               # http: openai | ollama
    api_key_env: Optional[str] = None  # http: NAME of the env var holding the key
    version: Optional[str] = None
    timeout: float = 30.0
    max_concurrency: int = 4          # simultaneous encoder calls, all tenants


@dataclass
class ReplicaConfig:
    source: str = ""                  # "http://primary:8080" or "dir:/path/to/primary/data"
    sync_interval_seconds: float = 15.0
    max_staleness_seconds: float = 120.0   # refuse reads when the last good sync is older
    fetch_timeout_seconds: float = 30.0
    max_file_bytes: int = 2 * 1024 ** 3
    autostart: bool = True                 # run the background sync loop (tests drive sync_once() by hand)


@dataclass
class RouterConfig:
    #: [{"name": "s0", "primary": "http://host:8080", "replicas": ["http://host:8081"]}, ...]
    shards: list = field(default_factory=list)
    #: tenant id -> shard name, overriding the hash (used while moving a tenant by hand)
    pins: dict = field(default_factory=dict)
    read_preference: str = "primary"          # primary | replica_preferred
    upstream_timeout_seconds: float = 30.0
    expose_backend_header: bool = False       # add X-Igv-Backend: <shard>/<primary|replica>


@dataclass
class ServerConfig:
    role: str = "primary"             # primary | replica
    data_dir: str = "./igv-data"
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "INFO"

    # connection / request handling
    max_workers: int = 32             # worker threads == max simultaneous connections served
    accept_queue: int = 64            # accepted-but-waiting connections before 503
    socket_timeout_seconds: float = 10.0   # idle / slow-read limit per socket operation
    max_body_bytes: int = 1_048_576
    max_header_bytes: int = 16_384
    max_requests_per_connection: int = 1000
    max_concurrent_requests: int = 32  # requests inside handlers (all tenants)
    queue_wait_seconds: float = 0.5   # wait for a concurrency slot before 503
    request_deadline_seconds: float = 30.0
    lock_timeout_seconds: float = 10.0  # wait for a graph lock before 503
    failed_auth_per_minute: int = 30   # per peer address, before 429 on bad keys
    registry_recheck_seconds: float = 1.0   # how soon a revoked key / disabled tenant takes effect
    max_page: int = 1000
    max_response_bytes: int = 16 * 1024 * 1024
    # Peer addresses allowed to tell us the real client via X-Forwarded-For
    # (the router, a reverse proxy). Used only for the failed-auth throttle.
    trusted_proxies: list = field(default_factory=list)

    # persistence
    durability: str = "snapshot"      # snapshot (periodic) | sync (save before every ack)
    autosave_interval_seconds: float = 30.0
    keep_generations: int = 3
    strict_startup: bool = False      # refuse to start when a graph fails verification

    quotas: Quotas = field(default_factory=Quotas)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    replica: ReplicaConfig = field(default_factory=ReplicaConfig)
    router: RouterConfig = field(default_factory=RouterConfig)

    # secrets, from the environment only (never read from the file)
    metrics_token: Optional[str] = field(default=None, repr=False)
    replication_token: Optional[str] = field(default=None, repr=False)

    def validate(self) -> "ServerConfig":
        if self.role not in ("primary", "replica"):
            raise ConfigError("role must be 'primary' or 'replica'")
        if self.durability not in ("snapshot", "sync"):
            raise ConfigError("durability must be 'snapshot' or 'sync'")
        if self.role == "replica" and self.durability == "sync":
            self.durability = "snapshot"
        if not (0 <= self.port <= 65535):
            raise ConfigError("port out of range")
        for name in ("max_workers", "accept_queue", "max_body_bytes", "max_header_bytes",
                     "max_concurrent_requests", "keep_generations", "max_page",
                     "max_requests_per_connection", "failed_auth_per_minute", "max_response_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"{name} must be an integer >= 1")
        for name in ("socket_timeout_seconds", "queue_wait_seconds",
                     "request_deadline_seconds", "lock_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ConfigError(f"{name} must be a positive number")
        if self.autosave_interval_seconds < 0:
            raise ConfigError("autosave_interval_seconds must be >= 0 (0 disables autosave)")
        if self.keep_generations < 2:
            raise ConfigError("keep_generations must be >= 2 (the previous one is the fallback)")
        if self.encoder.kind not in ("none", "mock", "http", "sentence-transformers"):
            raise ConfigError("encoder.kind must be none, mock, http or sentence-transformers")
        if self.encoder.kind == "mock" and not self.encoder.dimension:
            raise ConfigError("encoder.dimension is required for the mock encoder")
        if self.encoder.kind == "http" and not self.encoder.url:
            raise ConfigError("encoder.url is required for the http encoder")
        if self.encoder.kind in ("http",) and not self.encoder.model:
            raise ConfigError("encoder.model is required for the http encoder")
        if self.role == "replica" and not self.replica.source:
            raise ConfigError("replica.source is required when role is 'replica'")
        for tok_name in ("metrics_token", "replication_token"):
            tok = getattr(self, tok_name)
            if tok is not None and len(tok) < 24:
                raise ConfigError(f"{tok_name} (IGV_{tok_name.upper()}) must be at least 24 characters")
        self.quotas.checked()
        if self.router.read_preference not in ("primary", "replica_preferred"):
            raise ConfigError("router.read_preference must be 'primary' or 'replica_preferred'")
        seen = set()
        for sh in self.router.shards:
            if (not isinstance(sh, dict) or not isinstance(sh.get("name"), str)
                    or not isinstance(sh.get("primary"), str)
                    or not isinstance(sh.get("replicas", []), list)):
                raise ConfigError("each router shard needs a name, a primary URL and a replicas list")
            if sh["name"] in seen:
                raise ConfigError("duplicate shard name")
            seen.add(sh["name"])
        for tid, shard in self.router.pins.items():
            if shard not in seen:
                raise ConfigError(f"pin for {tid!r} names an unknown shard")
        return self


def _coerce(target: Any, data: Mapping[str, Any], where: str) -> None:
    fields = {f.name: f for f in dataclasses.fields(target)}
    for key, value in data.items():
        if key not in fields:
            raise ConfigError(f"unknown config key {where}{key!r}")
        if key in ("metrics_token", "replication_token"):
            raise ConfigError(f"{key} is a secret: set it via the IGV_{key.upper()} environment variable")
        current = getattr(target, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"{where}{key} must be an object")
            _coerce(current, value, f"{where}{key}.")
        else:
            setattr(target, key, value)


_ENV_SCALARS = {
    "IGV_ROLE": ("role", str),
    "IGV_DATA_DIR": ("data_dir", str),
    "IGV_HOST": ("host", str),
    "IGV_PORT": ("port", int),
    "IGV_LOG_LEVEL": ("log_level", str),
    "IGV_DURABILITY": ("durability", str),
    "IGV_AUTOSAVE_INTERVAL_SECONDS": ("autosave_interval_seconds", float),
    "IGV_MAX_WORKERS": ("max_workers", int),
    "IGV_MAX_BODY_BYTES": ("max_body_bytes", int),
    "IGV_STRICT_STARTUP": ("strict_startup", lambda s: s.lower() in ("1", "true", "yes")),
}


def load_config(path: Optional[str] = None,
                env: Optional[Mapping[str, str]] = None) -> ServerConfig:
    """Build a validated config from an optional JSON file plus ``IGV_*`` env vars."""
    env = os.environ if env is None else env
    cfg = ServerConfig()
    path = path or env.get("IGV_CONFIG")
    if path:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from None
        if not isinstance(data, dict):
            raise ConfigError("config file must contain a JSON object")
        _coerce(cfg, data, "")
    for var, (attr, conv) in _ENV_SCALARS.items():
        if var in env:
            try:
                setattr(cfg, attr, conv(env[var]))
            except ValueError:
                raise ConfigError(f"{var} has an invalid value") from None
    if env.get("IGV_REPLICA_SOURCE"):
        cfg.replica.source = env["IGV_REPLICA_SOURCE"]
    cfg.metrics_token = env.get("IGV_METRICS_TOKEN") or None
    cfg.replication_token = env.get("IGV_REPLICATION_TOKEN") or None
    return cfg.validate()
