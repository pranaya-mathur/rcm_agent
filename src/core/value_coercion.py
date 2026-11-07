"""
Strict value coercion helpers for API / payload normalization.

Avoid Python's truthiness traps (e.g. bool("false") is True).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple


TRUE_STRINGS = {"true", "1", "yes", "y"}
FALSE_STRINGS = {"false", "0", "no", "n"}


class CoercionError(ValueError):
    pass


def coerce_bool(value: Any, *, field: str = "value", default: Optional[bool] = None) -> bool:
    """
    Strict boolean normalization.
    Accepts: True/False, 0/1, "true"/"false"/"0"/"1"/"yes"/"no" (case-insensitive).
    Rejects: other types/strings (does not use Python bool()).
    """
    if value is None:
        if default is not None:
            return default
        raise CoercionError(f"{field} is required")

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1 or value == 1.0:
            return True
        if value == 0 or value == 0.0:
            return False
        raise CoercionError(f"{field} must be boolean-like (got numeric {value})")

    if isinstance(value, str):
        v = value.strip().lower()
        if v in TRUE_STRINGS:
            return True
        if v in FALSE_STRINGS:
            return False
        raise CoercionError(f"{field} must be boolean-like (got string {value!r})")

    raise CoercionError(f"{field} must be boolean-like (got type {type(value).__name__})")


def try_coerce_bool(value: Any, *, field: str = "value", default: Optional[bool] = None) -> Tuple[Optional[bool], Optional[str]]:
    try:
        return coerce_bool(value, field=field, default=default), None
    except CoercionError as e:
        return None, str(e)


def coerce_finite_number(
    value: Any,
    *,
    field: str = "value",
    allow_negative: bool = False,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Parse a finite float; optionally disallow negatives / bound range."""
    if value is None or value == "":
        raise CoercionError(f"{field} is required")
    try:
        f = float(value)
    except Exception as e:
        raise CoercionError(f"{field} must be numeric") from e
    if not math.isfinite(f):
        raise CoercionError(f"{field} must be a finite number (got {value!r})")
    if not allow_negative and f < 0:
        raise CoercionError(f"{field} must be >= 0")
    if minimum is not None and f < minimum:
        raise CoercionError(f"{field} must be >= {minimum}")
    if maximum is not None and f > maximum:
        raise CoercionError(f"{field} must be <= {maximum}")
    return f
