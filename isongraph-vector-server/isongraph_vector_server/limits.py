"""Rate limiting, bounded concurrency and the graph read/write lock."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Callable, Deque, Dict, Optional

from .errors import Overloaded, RateLimited


class TokenBucket:
    """Classic token bucket. ``take(cost)`` returns 0.0 on success, else the
    number of seconds until ``cost`` tokens will be available."""

    def __init__(self, rate: float, burst: int, clock: Callable[[], float] = time.monotonic):
        self.rate = float(rate)
        self.burst = float(burst)
        self._clock = clock
        self._tokens = float(burst)
        self._stamp = clock()
        self._lock = threading.Lock()

    def take(self, cost: float = 1.0) -> float:
        cost = min(cost, self.burst)   # a request bigger than the bucket is charged the bucket
        with self._lock:
            now = self._clock()
            self._tokens = min(self.burst, self._tokens + (now - self._stamp) * self.rate)
            self._stamp = now
            if self._tokens >= cost:
                self._tokens -= cost
                return 0.0
            return (cost - self._tokens) / self.rate

    def reconfigure(self, rate: float, burst: int) -> None:
        with self._lock:
            self.rate, self.burst = float(rate), float(burst)
            self._tokens = min(self._tokens, self.burst)


class SlidingWindow:
    """Count of events per key in the last ``window`` seconds (failed-auth throttle)."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic,
                 max_keys: int = 10_000):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._events: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def blocked(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            q = self._events.get(key)
            if not q:
                return False
            while q and now - q[0] > self.window:
                q.popleft()
            return len(q) >= self.limit

    def record(self, key: str) -> None:
        now = self._clock()
        with self._lock:
            if len(self._events) >= self._max_keys and key not in self._events:
                # Bounded memory under an address-spraying attack: drop the oldest key.
                self._events.pop(next(iter(self._events)))
            q = self._events.setdefault(key, deque())
            q.append(now)
            while q and now - q[0] > self.window:
                q.popleft()


class Gate:
    """A counting semaphore that gives up after ``wait`` seconds instead of queueing forever."""

    def __init__(self, size: int):
        self.size = size
        self._sem = threading.BoundedSemaphore(size)
        self._lock = threading.Lock()
        self.in_flight = 0

    @contextmanager
    def enter(self, wait: float, error: Optional[Exception] = None):
        if not self._sem.acquire(timeout=wait):
            raise error or Overloaded("server is at its concurrency limit; retry shortly",
                                      headers={"Retry-After": "1"})
        with self._lock:
            self.in_flight += 1
        try:
            yield
        finally:
            with self._lock:
                self.in_flight -= 1
            self._sem.release()


class LockTimeout(Exception):
    pass


class RWLock:
    """Readers-writer lock; writers are preferred so a stream of searches cannot
    starve an upsert. Acquisition takes a timeout and raises :class:`LockTimeout`.

    Not re-entrant: a thread must not take it twice.
    """

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    def acquire_read(self, timeout: Optional[float] = None) -> None:
        end = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._writer or self._writers_waiting:
                if not self._wait(end):
                    raise LockTimeout()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self, timeout: Optional[float] = None) -> None:
        end = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers:
                    if not self._wait(end):
                        raise LockTimeout()
            finally:
                self._writers_waiting -= 1
                if self._writers_waiting == 0 and not self._writer:
                    self._cond.notify_all()
            self._writer = True

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()

    def _wait(self, end: Optional[float]) -> bool:
        if end is None:
            self._cond.wait()
            return True
        remaining = end - time.monotonic()
        if remaining <= 0:
            return False
        self._cond.wait(remaining)
        return True

    @contextmanager
    def read(self, timeout: Optional[float] = None):
        self.acquire_read(timeout)
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write(self, timeout: Optional[float] = None):
        self.acquire_write(timeout)
        try:
            yield
        finally:
            self.release_write()


def rate_limited(retry_after: float) -> RateLimited:
    secs = max(1, int(retry_after + 0.999))
    return RateLimited("rate limit exceeded", headers={"Retry-After": str(secs)})
