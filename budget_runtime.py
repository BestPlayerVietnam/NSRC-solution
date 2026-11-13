"""CPU-only accounting for one contest budget; no networking or implicit retries.

Amounts are integer decitokens: one input token costs one unit and one output
token (including reasoning) costs ten. The caller must disable SDK retries and
pass Reservation.timeout to its transport/asyncio deadline guard.

The reservation formula is deliberately conservative, not a tokenizer theorem:
2 * output_cap + UTF-8 prompt bytes / 10 + 128 weighted tokens. It assumes the
provider enforces the supplied cap; its reasoning-cap semantics must be checked.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Callable


class BudgetExhausted(RuntimeError):
    pass


class DeadlineExceeded(BudgetExhausted):
    pass


_MISSING = object()


def _field(obj: object, key: str) -> object:
    if isinstance(obj, Mapping):
        return obj.get(key, _MISSING)
    return getattr(obj, key, _MISSING)


def _counter(value: object) -> int | None:
    # A malformed/string/defaulted counter must not release a reservation.
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _aliases(raw: object, names: tuple[str, ...]) -> int | None:
    values = [_field(raw, key) for key in names]
    values = [value for value in values if value is not _MISSING and value is not None]
    if not values:
        return None
    parsed = [_counter(value) for value in values]
    if any(value is None for value in parsed) or len(set(parsed)) != 1:
        return None
    return parsed[0]


def usage_snapshot(raw: object) -> dict:
    """Copy raw accounting fields only, never an exception, prompt or headers."""
    result = {}
    for key in ("input_tokens", "prompt_tokens", "output_tokens", "completion_tokens",
                "reasoning_tokens", "total_tokens", "cache_tokens"):
        value = _field(raw, key)
        if value is not _MISSING:
            result[key] = value if value is None or isinstance(value, (int, float, bool)) else "invalid"
    for key in ("completion_tokens_details", "output_tokens_details", "prompt_tokens_details",
                "input_tokens_details"):
        detail = _field(raw, key)
        if detail is _MISSING or detail is None:
            continue
        result[key] = {}
        for name in ("reasoning_tokens", "cached_tokens", "audio_tokens",
                     "accepted_prediction_tokens", "rejected_prediction_tokens"):
            value = _field(detail, name)
            if value is not _MISSING:
                result[key][name] = value if value is None or isinstance(value, (int, float, bool)) else "invalid"
    return result


@dataclass(frozen=True)
class UsageCharge:
    input_tokens: int
    output_tokens: int  # All billed output, including reasoning exactly once.
    reasoning_tokens: int
    units: int
    mode: str

    @property
    def weighted_tokens(self) -> float:
        return self.units / 10


def normalize_usage(raw: object) -> UsageCharge | None:
    """Trust complete, internally consistent provider counters only.

    input+output==total proves inclusive output; input+output+reasoning==total
    proves separately reported reasoning. Conflicting/missing totals, negative
    values and all-zero SDK defaults retain the full reservation instead.
    Cache counters intentionally never reduce input cost.
    """
    if raw is None:
        return None
    inp = _aliases(raw, ("input_tokens", "prompt_tokens"))
    out = _aliases(raw, ("output_tokens", "completion_tokens"))
    total = _aliases(raw, ("total_tokens",))
    reasoning_values = []
    direct = _field(raw, "reasoning_tokens")
    if direct is not _MISSING and direct is not None:
        reasoning_values.append(direct)
    for key in ("completion_tokens_details", "output_tokens_details"):
        details = _field(raw, key)
        value = _field(details, "reasoning_tokens")
        if value is not _MISSING and value is not None:
            reasoning_values.append(value)
    reasoning = [_counter(value) for value in reasoning_values]
    if any(value is None for value in reasoning) or len(set(reasoning)) > 1:
        return None
    reason = reasoning[0] if reasoning else 0
    if inp is None or out is None or total is None or total == 0:
        return None
    if total == inp + out and reason <= out:
        billed = out
        mode = "reasoning_included" if reason else "output_total"
    elif reason > 0 and total == inp + out + reason:
        billed = out + reason
        mode = "reasoning_separate"
    else:
        return None
    return UsageCharge(inp, billed, reason, inp + 10 * billed, mode)


def _untrusted_usage_reserve(raw: object) -> int:
    """Do not ignore reported costs exceeding the pre-call assumption.

    This is a conservative reservation, never a trusted actual charge. With an
    inconsistent total, treating all total tokens as output avoids assuming an
    unknown input/output split. Never sum unproven separate reasoning here.
    """
    def maximum(obj: object, names: tuple[str, ...]) -> int:
        values = [_counter(_field(obj, name)) for name in names]
        return max((value for value in values if value is not None), default=0)
    inp = maximum(raw, ("input_tokens", "prompt_tokens"))
    out = maximum(raw, ("output_tokens", "completion_tokens", "reasoning_tokens"))
    for key in ("completion_tokens_details", "output_tokens_details"):
        out = max(out, maximum(_field(raw, key), ("reasoning_tokens",)))
    total = maximum(raw, ("total_tokens",))
    return max(inp + 10 * out, 10 * total)


@dataclass(frozen=True)
class Reservation:
    call_id: int
    model: str
    max_tokens: int
    reserved_units: int
    timeout: float

    @property
    def weighted_tokens(self) -> float:
        return self.reserved_units / 10


class BudgetLedger:
    """Thread-safe, single-process shared budget, including in-flight calls.

    Each prepare is one actual API attempt. reconcile or hold finalizes it once;
    neither method retries. Pending and unknown calls remain fully charged.
    A provider charge above its reservation is recorded in full and blocks new
    calls, allowing the caller to save the already returned answer first.
    """

    def __init__(self, budget: float = 100_000, *, clock: Callable[[], float] = time.monotonic):
        try:
            amount = Decimal(str(budget))
        except InvalidOperation as exc:
            raise ValueError("budget must be a finite nonnegative number") from exc
        if not amount.is_finite() or amount < 0:
            raise ValueError("budget must be a finite nonnegative number")
        self.limit_units = int((amount * 10).to_integral_value(rounding=ROUND_FLOOR))
        self._spent_units = 0
        self._clock = clock
        self._lock = threading.Lock()
        self._reservations: dict[int, Reservation] = {}
        self._records: dict[int, dict] = {}
        self._blocked = False

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent_units / 10

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0, self.limit_units - self._spent_units) / 10

    @property
    def blocked(self) -> bool:
        with self._lock:
            return self._blocked

    @property
    def call_records(self) -> list[dict]:
        # Return independent JSON-compatible records, not mutable ledger state.
        import copy
        with self._lock:
            return copy.deepcopy(list(self._records.values()))

    def prepare(self, system: str, user: str, *, max_tokens: int = 400,
                min_output_tokens: int = 32, model: str = "", deadline: float | None = None,
                timeout: float = 90.0, extra_input_bytes: int = 0) -> Reservation:
        """Reserve before sending, shrink cap if necessary, and bound timeout.

        Include any additional history/tool schema bytes in extra_input_bytes.
        deadline is an absolute time.monotonic deadline, not wall-clock time.
        """
        for name, value in (("max_tokens", max_tokens), ("min_output_tokens", min_output_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if max_tokens < min_output_tokens:
            raise BudgetExhausted("requested output cap is below the minimum")
        if isinstance(extra_input_bytes, bool) or not isinstance(extra_input_bytes, int) or extra_input_bytes < 0:
            raise ValueError("extra_input_bytes must be a nonnegative integer")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        input_bytes = len(system.encode("utf-8")) + len(user.encode("utf-8")) + extra_input_bytes
        overhead_units = input_bytes + 1280
        with self._lock:
            if self._blocked:
                raise BudgetExhausted("provider exceeded a reservation; accounting guard stopped new calls")
            if deadline is not None:
                timeout = min(timeout, deadline - self._clock())
            if timeout <= 0:
                raise DeadlineExceeded("request deadline has passed")
            available = self.limit_units - self._spent_units - overhead_units
            cap = min(max_tokens, available // 20)
            if cap < min_output_tokens:
                raise BudgetExhausted("local token budget guard")
            reserved = overhead_units + 20 * cap
            call_id = len(self._records) + 1
            reservation = Reservation(call_id, model, cap, reserved, timeout)
            self._spent_units += reserved
            self._reservations[call_id] = reservation
            self._records[call_id] = {
                "call_id": call_id, "model": model, "max_tokens": cap,
                "reserved_units": reserved, "reserved_tokens": reserved / 10,
                "weighted_tokens": reserved / 10, "charged_units": reserved,
                "status": "pending", "raw_usage": None, "timeout": timeout,
            }
            return reservation

    def _pending(self, reservation: Reservation) -> dict:
        if self._reservations.get(reservation.call_id) is not reservation:
            raise ValueError("reservation belongs to another ledger or was not prepared")
        record = self._records[reservation.call_id]
        if record["status"] != "pending":
            raise ValueError("reservation was already finalized")
        return record

    def reconcile(self, reservation: Reservation, raw_usage: object) -> dict:
        charge = normalize_usage(raw_usage)
        raw = usage_snapshot(raw_usage)
        with self._lock:
            record = self._pending(reservation)
            record["raw_usage"] = raw
            if charge is None:
                record["status"] = "reserved_missing_or_invalid_usage"
                reserve = max(reservation.reserved_units, _untrusted_usage_reserve(raw_usage))
                if reserve > reservation.reserved_units:
                    self._spent_units += reserve - reservation.reserved_units
                    self._blocked = True
                    record.update(status="untrusted_usage_exceeded_reservation",
                                  charged_units=reserve, weighted_tokens=reserve / 10)
            else:
                self._spent_units += charge.units - reservation.reserved_units
                record.update(status="provider_reconciled", charged_units=charge.units,
                              weighted_tokens=charge.weighted_tokens,
                              input_tokens=charge.input_tokens, output_tokens=charge.output_tokens,
                              reasoning_tokens=charge.reasoning_tokens, usage_mode=charge.mode)
                if charge.units > reservation.reserved_units:
                    self._blocked = True
                    record["status"] = "provider_exceeded_reservation"
            return dict(record)

    settle = reconcile

    def hold(self, reservation: Reservation, reason: str = "error") -> dict:
        # Never serialize raw exception messages, which can include credentials.
        safe_reason = reason if re.fullmatch(r"[a-z_]{1,40}", reason) else "error"
        with self._lock:
            record = self._pending(reservation)
            record["status"] = "reserved_" + safe_reason
            return dict(record)


@dataclass(frozen=True)
class ErrorDisposition:
    kind: str
    retryable: bool
    abort_run: bool
    status_code: int | None = None


def classify_api_error(exc: BaseException) -> ErrorDisposition:
    """Classify, never retry or release tokens. A 429 is not always exhaustion."""
    errors = []
    current: BaseException | None = exc
    for _ in range(5):
        if current is None or any(current is item for item in errors):
            break
        errors.append(current)
        current = current.__cause__ or current.__context__
    descriptions = [str(error).lower() for error in errors]
    for error in errors:
        code = getattr(error, "code", None)
        if isinstance(code, str):
            descriptions.append(code.lower())
        body = getattr(error, "body", None)
        if isinstance(body, Mapping):
            nested = body.get("error", body)
            if isinstance(nested, Mapping):
                for key in ("code", "type", "message"):
                    value = nested.get(key)
                    if isinstance(value, str):
                        descriptions.append(value.lower())
    message = " ".join(descriptions)
    status = None
    for error in errors:
        candidate = getattr(error, "status_code", None)
        if candidate is None:
            candidate = getattr(getattr(error, "response", None), "status_code", None)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            status = candidate
            break
    if status is None:
        match = re.search(r"\b(?:status(?:_code| code)?|error code|http)\s*[:=]?\s*(\d{3})\b", message)
        if match:
            status = int(match.group(1))
    quota_markers = ("insufficient_quota", "quota_exceeded", "billing_hard_limit_reached",
                     "insufficient credits", "insufficient balance", "credit balance is too low",
                     "budget exhausted", "quota exhausted", "exceeded your current quota")
    if status == 402 or any(marker in message for marker in quota_markers):
        return ErrorDisposition("quota", False, True, status)
    if status in (401, 403):
        return ErrorDisposition("authentication", False, True, status)
    if status == 429 or "rate_limit_exceeded" in message:
        return ErrorDisposition("rate_limit", True, False, status)
    if any(isinstance(error, TimeoutError) for error in errors):
        return ErrorDisposition("timeout", True, False, status)
    if status in (408, 409, 425, 500, 502, 503, 504):
        return ErrorDisposition("transient", True, False, status)
    if any(isinstance(error, ConnectionError) for error in errors):
        return ErrorDisposition("connection", True, False, status)
    return ErrorDisposition("other", False, False, status)
