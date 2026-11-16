"""Assemble and run a node (primary or replica) or a router."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from isongraph_vector import EmbeddingEncoder

from .auth import TenantRegistry
from .config import ServerConfig
from .encoders import BoundedEncoder, build_encoder
from .engine import Engine
from .httpd import BoundedHTTPServer
from .logs import log, setup_logging
from .metrics import Metrics
from .replication import ReplicaSyncer
from .router import Router
from .service import App


class Server:
    """A storage node. ``start()`` returns once it is listening and ready."""

    def __init__(self, cfg: ServerConfig, encoder: Optional[EmbeddingEncoder] = None,
                 replica_source=None, configure_logging: bool = True):
        self.cfg = cfg.validate()
        if configure_logging:
            setup_logging(cfg.log_level)
        self.metrics = Metrics()
        if encoder is not None and not isinstance(encoder, BoundedEncoder):
            encoder = BoundedEncoder(encoder, cfg.encoder.max_concurrency)
        self.encoder = encoder if encoder is not None else build_encoder(cfg.encoder)
        if self.encoder is not None:
            # Learn an HTTP encoder's dimension now; an unreachable endpoint is logged, not fatal
            # (graph creation and text requests then answer 503 encoder_failed until it is back).
            try:
                _ = self.encoder.dimension
            except Exception as exc:
                log(logging.WARNING, "embedding encoder not reachable at startup", error=type(exc).__name__)
        self.engine = Engine(cfg, self.encoder, self.metrics)
        self.registry = TenantRegistry(cfg.data_dir, cfg.quotas, cfg.registry_recheck_seconds)
        self.replica: Optional[ReplicaSyncer] = None
        self._replica_source = replica_source
        self.app: Optional[App] = None
        self.httpd: Optional[BoundedHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self):
        return self.httpd.server_address[:2]

    @property
    def url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

    def start(self) -> "Server":
        self.engine.start()
        self.registry.rescan(force=True)
        if self.cfg.role == "replica":
            self.replica = ReplicaSyncer(self.cfg, self.engine, self.registry, self.metrics,
                                         source=self._replica_source)
        self.app = App(self.cfg, self.engine, self.registry, self.metrics, self.replica)
        self.httpd = BoundedHTTPServer((self.cfg.host, self.cfg.port), self.app, self.cfg)
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.1},
                                        name="igv-accept", daemon=True)
        self._thread.start()
        if self.replica is not None and self.cfg.replica.autostart:
            self.replica.start()
        log(logging.INFO, "listening", role=self.cfg.role, host=self.address[0], port=self.address[1],
            graphs_loaded=self.engine.startup_report["loaded"],
            graphs_quarantined=self.engine.startup_report["quarantined"])
        return self

    def stop(self) -> None:
        """Graceful shutdown: stop accepting, finish in-flight requests, final snapshot."""
        if self.replica is not None:
            self.replica.stop()
        if self.httpd is not None:
            self.httpd.stop()
        self.engine.shutdown()
        log(logging.INFO, "stopped")

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()


class RouterServer:
    def __init__(self, cfg: ServerConfig, configure_logging: bool = True):
        self.cfg = cfg.validate()
        if configure_logging:
            setup_logging(cfg.log_level)
        self.router = Router(cfg)
        self.httpd: Optional[BoundedHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self):
        return self.httpd.server_address[:2]

    @property
    def url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

    def start(self) -> "RouterServer":
        self.httpd = BoundedHTTPServer((self.cfg.host, self.cfg.port), self.router, self.cfg)
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.1},
                                        name="igv-router-accept", daemon=True)
        self._thread.start()
        log(logging.INFO, "router listening", host=self.address[0], port=self.address[1],
            shards=len(self.cfg.router.shards))
        return self

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.stop()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
