"""The error model.

Every failure a client can see is an :class:`ApiError`, rendered as

    {"error": {"code": "...", "message": "...", "request_id": "...", "details": [...]}}

Anything that is not an ApiError is logged server-side (with a traceback) and
answered with the generic ``internal_error``: no exception text, no stack
trace and no file path ever reaches a client.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ApiError(Exception):
    status = 500
    code = "internal_error"

    def __init__(self, message: str = "", *, code: Optional[str] = None,
                 status: Optional[int] = None,
                 details: Optional[List[Dict[str, Any]]] = None,
                 headers: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.message = message or self.code
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        self.details = details
        self.headers = headers or {}

    def to_body(self, request_id: str) -> Dict[str, Any]:
        err: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "request_id": request_id,
        }
        if self.details:
            err["details"] = self.details
        return {"error": err}


class BadRequest(ApiError):
    status = 400
    code = "invalid_request"


class Unauthorized(ApiError):
    status = 401
    code = "unauthorized"


class Forbidden(ApiError):
    status = 403
    code = "forbidden"


class NotFound(ApiError):
    status = 404
    code = "not_found"


class MethodNotAllowed(ApiError):
    status = 405
    code = "method_not_allowed"


class Conflict(ApiError):
    status = 409
    code = "conflict"


class LengthRequired(ApiError):
    status = 411
    code = "length_required"


class PayloadTooLarge(ApiError):
    status = 413
    code = "payload_too_large"


class UnsupportedMediaType(ApiError):
    status = 415
    code = "unsupported_media_type"


class QuotaExceeded(ApiError):
    status = 403
    code = "quota_exceeded"


class RateLimited(ApiError):
    status = 429
    code = "rate_limited"


class Overloaded(ApiError):
    status = 503
    code = "overloaded"


class Unavailable(ApiError):
    status = 503
    code = "unavailable"


class DeadlineExceeded(ApiError):
    status = 504
    code = "deadline_exceeded"


class BadGateway(ApiError):
    status = 502
    code = "bad_gateway"


def validation_error(path: str, message: str) -> BadRequest:
    """A schema violation. ``path`` is a JSONPath-ish pointer into the body."""
    return BadRequest(
        f"{path}: {message}",
        code="validation_failed",
        details=[{"path": path, "message": message}],
    )
