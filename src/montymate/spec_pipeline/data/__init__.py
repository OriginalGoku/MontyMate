"""Data structures for the spec_pipeline.

These modules contain *pure-ish* data + normalization helpers, and (optionally) very light
persistence adapters (e.g., file-backed SpecStore) so components can be unit-tested
without involving the DB/UI.
"""

from .spec_types import Spec
from .human_inputs import QuestionAnswer, ClarificationBatch, normalize_answers
from .spec_store import SpecStore, SpecStorePaths, FileSpecStore, MemorySpecStore

__all__ = [
    "Spec",
    "SpecDict",
    "SPEC_KEYS",
    "HumanAnswer",
    "HumanAnswerBatch",
    "normalize_answers",
    "SpecStore",
    "SpecStorePaths",
    "FileSpecStore",
    "MemorySpecStore",
]
