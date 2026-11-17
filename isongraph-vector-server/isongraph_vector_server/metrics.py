"""Aggregate metrics in Prometheus text format.

Deliberately has no tenant or graph labels: the global ``/metrics`` endpoint
is protected by an operator token, but even so a per-tenant label would put
tenant ids (and by extension their activity) into whatever scrapes it. Per
tenant numbers are available to that tenant itself on ``GET /v1/usage``.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self._gauges: Dict[str, float] = {}
        self._hist: Dict[str, list] = {}   # route -> [bucket counts..., +Inf, sum, count]

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, route: str, seconds: float) -> None:
        with self._lock:
            h = self._hist.get(route)
            if h is None:
                h = self._hist[route] = [0] * (len(BUCKETS) + 1) + [0.0, 0]
            for i, bound in enumerate(BUCKETS):
                if seconds <= bound:
                    h[i] += 1
            h[len(BUCKETS)] += 1
            h[-2] += seconds
            h[-1] += 1

    def counter(self, name: str, **labels: str) -> float:
        with self._lock:
            return self._counters.get((name, tuple(sorted(labels.items()))), 0.0)

    def render(self) -> str:
        lines = []
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            hist = {k: list(v) for k, v in self._hist.items()}
        seen = set()
        for (name, labels), value in sorted(counters.items()):
            if name not in seen:
                lines.append(f"# TYPE {name} counter")
                seen.add(name)
            lab = ",".join(f'{k}="{_esc(v)}"' for k, v in labels)
            lines.append(f"{name}{{{lab}}} {value:g}" if lab else f"{name} {value:g}")
        for name, value in sorted(gauges.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value:g}")
        if hist:
            lines.append("# TYPE igv_request_seconds histogram")
        for route, h in sorted(hist.items()):
            r = _esc(route)
            for i, bound in enumerate(BUCKETS):
                lines.append(f'igv_request_seconds_bucket{{route="{r}",le="{bound:g}"}} {h[i]}')
            lines.append(f'igv_request_seconds_bucket{{route="{r}",le="+Inf"}} {h[len(BUCKETS)]}')
            lines.append(f'igv_request_seconds_sum{{route="{r}"}} {h[-2]:g}')
            lines.append(f'igv_request_seconds_count{{route="{r}"}} {h[-1]}')
        return "\n".join(lines) + "\n"


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
