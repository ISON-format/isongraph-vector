"""Structured (JSON-lines) logging that cannot carry secrets.

Two layers: the request path never hands a header or body to the logger (it
logs method, route *template*, status, latency, tenant id and request id),
and a filter redacts anything shaped like an API key or bearer credential
from every record, including exception text, as a backstop.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from typing import Any

_KEY = re.compile(r"igv_[a-z0-9-]{2,32}\.[0-9a-f]{16}\.[A-Za-z0-9_-]{20,}")
_BEARER = re.compile(r"(?i)(bearer|authorization[\"']?\s*[:=])\s*[^\s,;\"']+")

LOGGER_NAME = "igv"


def redact(text: str) -> str:
    text = _KEY.sub("igv_[REDACTED]", text)
    return _BEARER.sub(lambda m: m.group(1) + " [REDACTED]", text)


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.getMessage()))
        record.args = ()
        fields = getattr(record, "fields", None)
        if fields:
            record.fields = {k: redact(v) if isinstance(v, str) else v for k, v in fields.items()}
        if record.exc_info and record.exc_info[1] is not None:
            # Render the traceback ourselves so it passes through redaction too.
            import traceback
            record.exc_text = redact("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": round(time.time(), 3),
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
        }
        out.update(getattr(record, "fields", {}) or {})
        if record.exc_text:
            out["exc"] = record.exc_text
        return json.dumps(out, separators=(",", ":"), default=str)


def setup_logging(level: str = "INFO", stream: Any = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    return logger


def log(level: int, msg: str, **fields: Any) -> None:
    logging.getLogger(LOGGER_NAME).log(level, msg, extra={"fields": fields})
