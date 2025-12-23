from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OutputFormat(str, Enum):
    """Defines the serialization format a component is expected to output.

    The enum is used to enforce consistent naming across components and to
    support validation and parsing logic in a centralized way.

    Values:
    - YAML: Indicates the component must return YAML text (no markdown fences).
    - JSON: Indicates the component must return JSON text (no markdown fences).
    - TEXT: Indicates the component may return plain text without a structured schema.
    """

    YAML = "yaml"
    JSON = "json"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class ComponentMetadata:
    """Stores the stable identity and behavioral expectations of a component.

    The metadata is intended to be shared across tools and prompt builders to:
    - provide a consistent human-readable description of the component’s purpose
    - declare the required output format for parsing/validation
    - attach lightweight tags for routing, filtering, logging, or UI grouping

    Fields:
    - name: Canonical component name used in logs, traces, and registries.
    - output_format: Declares the structured output format expected from the component.
    - description: Short, plain-language statement of what the component does.
    - tags: Optional labels used for grouping or capability hints (e.g. "spec", "critic").
    """

    name: str
    output_format: OutputFormat
    description: str
    tags: tuple[str, ...] = ()
