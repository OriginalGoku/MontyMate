from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from montymate.events import ActorType, EventType, EventSpec, EVENT_SPECS, Severity


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: Sequence[str] = ()


def _missing_keys(payload: Mapping[str, Any], required: Sequence[str]) -> list[str]:
    return [k for k in required if k not in payload]


def validate_event_envelope(
    *,
    event_type: str,
    actor_type: str,
    severity: str,
    payload: Mapping[str, Any],
    strict_event_types: bool = True,
) -> None:
    """
    Validates:
    - event_type exists (if strict)
    - actor_type and severity are valid enums
    - payload contains required keys per event type
    """
    errors: list[str] = []

    # actor_type
    try:
        ActorType(actor_type)
    except ValueError:
        errors.append(f"Invalid actor_type='{actor_type}'. Allowed: {[a.value for a in ActorType]}")

    # severity
    try:
        Severity(severity)
    except ValueError:
        errors.append(f"Invalid severity='{severity}'. Allowed: {[s.value for s in Severity]}")

    # event type + payload contract
    spec: EventSpec | None = None
    try:
        et = EventType(event_type)
        spec = EVENT_SPECS.get(et)
        if spec is None:
            errors.append(f"No payload spec found for event_type='{event_type}'. Add it to EVENT_SPECS.")
    except ValueError:
        if strict_event_types:
            errors.append(f"Unknown event_type='{event_type}'. Add it to EventType.")
        else:
            spec = None  # unknown types allowed

    # payload required keys
    if spec is not None:
        missing = _missing_keys(payload, spec.required_keys)
        if missing:
            errors.append(f"Payload missing required keys for '{event_type}': {missing}")

        if not spec.allow_extra_keys:
            allowed = set(spec.required_keys) | set(spec.optional_keys)
            extras = [k for k in payload.keys() if k not in allowed]
            if extras:
                errors.append(f"Payload has extra keys not allowed for '{event_type}': {extras}")

    if errors:
        raise ValidationError("; ".join(errors))


def validate_event_payload_size_hint(
    *,
    payload_json_len: int,
    inline_max_bytes: int,
    warn_threshold_ratio: float = 0.9,
) -> ValidationResult:
    """
    Not a hard validation: just a hint that payload is approaching inline limits.
    Helpful during development to catch accidental giant payloads in the event stream.

    The EventLogger already handles splitting to artifacts; this is about developer feedback.
    """
    warn_at = int(inline_max_bytes * warn_threshold_ratio)
    if payload_json_len >= warn_at:
        return ValidationResult(ok=True, errors=(f"payload_json_len={payload_json_len} near inline_max_bytes={inline_max_bytes}",))
    return ValidationResult(ok=True)