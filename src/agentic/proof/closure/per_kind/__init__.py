"""Per-kind Γ closure rules — one module per :class:`ObligationKind`."""
from agentic.proof.closure.per_kind.argmax import gamma_argmax
from agentic.proof.closure.per_kind.count import gamma_count
from agentic.proof.closure.per_kind.exists import gamma_exists
from agentic.proof.closure.per_kind.forall import gamma_forall
from agentic.proof.closure.per_kind.negation import gamma_negation
from agentic.proof.closure.per_kind.set import gamma_set

__all__ = [
    "gamma_argmax",
    "gamma_count",
    "gamma_exists",
    "gamma_forall",
    "gamma_negation",
    "gamma_set",
]
