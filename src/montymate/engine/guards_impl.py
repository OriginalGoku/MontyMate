from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from montymate.core.types import JSON
from montymate.engine.guards import GuardEvaluator
from montymate.engine.state import GuardContext


def _get_by_dotted_path(obj: Any, dotted: str) -> Any:
    """
    Traverse nested dicts/lists using dotted paths like:
      - "decision_record.change_class"
      - "module_spec.intent.kind"
      - "artifacts.module_spec" (artifact logical name lookup)

    If the path can't be resolved, returns None.
    """
    if dotted == "":
        return obj

    cur: Any = obj
    for part in dotted.split("."):
        if cur is None:
            return None

        # list indexing support: "items.0.name"
        if isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
            continue

        if isinstance(cur, dict):
            cur = cur.get(part)
            continue

        return None

    return cur


def _resolve_ref(ref: str, ctx: GuardContext) -> Any:
    """
    Ref grammar (MVP):
      - "decision_record.<path>"
      - "artifacts.<logical_name>"
      - "run_id"
      - "step_id"
    """
    if ref == "run_id":
        return ctx.run_id
    if ref == "step_id":
        return ctx.step_id

    if ref.startswith("decision_record."):
        return _get_by_dotted_path(ctx.decision_record, ref[len("decision_record.") :])

    if ref.startswith("artifacts."):
        logical = ref[len("artifacts.") :]
        return ctx.artifacts.get(logical)

    # unknown refs resolve to None (guard can handle via exists/is_null checks)
    return None


def _value(node: Any, ctx: GuardContext) -> Any:
    """
    Convert a node into a value.
    Supported forms:
      - {"ref": "decision_record.change_class"}
      - {"literal": 123}
      - any primitive (str/int/bool/list/dict) treated as a literal
    """
    if isinstance(node, dict):
        if "ref" in node:
            return _resolve_ref(str(node["ref"]), ctx)
        if "literal" in node:
            return node["literal"]
    return node


@dataclass(frozen=True)
class DefaultGuardEvaluator(GuardEvaluator):
    """
    Minimal structured guard evaluator for MVP.
    No string-eval; all logic is explicit and auditable.
    """

    def eval(self, guard_obj: JSON, ctx: GuardContext) -> bool:
        if not isinstance(guard_obj, dict):
            raise ValueError(f"Guard must be an object; got: {type(guard_obj).__name__}")

        op = guard_obj.get("op")
        if not op:
            raise ValueError("Guard object missing 'op'")

        # --------- trivial ops ----------
        if op == "always_true":
            return True
        if op == "always_false":
            return False

        # --------- boolean composition ----------
        if op == "not":
            inner = guard_obj.get("guard")
            if inner is None:
                raise ValueError("not-guard missing 'guard'")
            return not self.eval(inner, ctx)

        if op in ("and", "or"):
            guards = guard_obj.get("guards")
            if not isinstance(guards, list) or not guards:
                raise ValueError(f"{op}-guard missing non-empty 'guards' list")
            results = [self.eval(g, ctx) for g in guards]
            return all(results) if op == "and" else any(results)

        # --------- comparisons ----------
        if op in ("eq", "ne", "lt", "lte", "gt", "gte"):
            a = _value(guard_obj.get("a"), ctx)
            b = _value(guard_obj.get("b"), ctx)

            if op == "eq":
                return a == b
            if op == "ne":
                return a != b
            if op == "lt":
                return a < b
            if op == "lte":
                return a <= b
            if op == "gt":
                return a > b
            if op == "gte":
                return a >= b

        # --------- membership / contains ----------
        if op == "in":
            item = _value(guard_obj.get("item"), ctx)
            container = _value(guard_obj.get("container"), ctx)
            try:
                return item in container  # type: ignore[operator]
            except TypeError:
                return False

        if op == "contains":
            container = _value(guard_obj.get("container"), ctx)
            item = _value(guard_obj.get("item"), ctx)
            try:
                return item in container  # type: ignore[operator]
            except TypeError:
                return False

        # --------- existence / null checks ----------
        if op == "exists":
            v = _value(guard_obj.get("value"), ctx)
            return v is not None

        if op == "is_null":
            v = _value(guard_obj.get("value"), ctx)
            return v is None

        if op == "truthy":
            v = _value(guard_obj.get("value"), ctx)
            return bool(v)

        if op == "falsy":
            v = _value(guard_obj.get("value"), ctx)
            return not bool(v)

        raise ValueError(f"Unsupported guard op '{op}'")