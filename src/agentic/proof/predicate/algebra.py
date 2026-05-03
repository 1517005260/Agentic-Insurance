"""AND-canonicalisation, serialisation and entailment.

Entailment is **syntactic only** (matches the design doc): two specs
entail each other iff they share the same canonical hash, with AND-set
containment as the only composition rule. This keeps closure decidable
and avoids the PSPACE-hardness of regex subsumption.
"""
from typing import Any, List, Tuple

from agentic.proof.predicate.registry import PredicateError, _PRIMITIVES
from agentic.proof.types import PredicateSpec


def build_and_spec(conjuncts: List[PredicateSpec]) -> PredicateSpec:
    """Compose conjuncts into a canonical ``and_`` PredicateSpec.

    Conjuncts are sorted by their serialized hash so two specs with the
    same set of children hash equal regardless of input order. Empty
    conjunct lists are rejected as universal.
    """
    if not conjuncts:
        raise PredicateError("and_() requires at least one conjunct.")
    if any(c.name == "and" for c in conjuncts):
        # Flatten nested ANDs to keep canonical form trivial.
        flat: List[PredicateSpec] = []
        for c in conjuncts:
            if c.name == "and":
                child_specs = [PredicateSpec(name=cc[0], args=cc[1]) for cc in c.args]
                flat.extend(child_specs)
            else:
                flat.append(c)
        conjuncts = flat
    serialized = [serialize_spec(c) for c in conjuncts]
    sorted_pairs = sorted(zip(serialized, conjuncts), key=lambda p: p[0])
    args: List[Tuple[str, Any]] = []
    for _, c in sorted_pairs:
        args.append((c.name, c.args))
    return PredicateSpec(name="and", args=tuple(args))


def serialize_spec(spec: PredicateSpec) -> str:
    """Stable canonical hash string for a spec.

    Used as the key for entailment equality checks. Format is intended
    to be readable in logs (e.g.
    ``and(contains_string{pattern=foo}, regex_match{pattern=^X})``).
    """
    if spec.name == "and":
        children = []
        for child_name, child_args in spec.args:
            child_spec = PredicateSpec(name=child_name, args=child_args)
            children.append(serialize_spec(child_spec))
        return f"and({','.join(children)})"
    items = ",".join(f"{k}={v!r}" for k, v in spec.args)
    return f"{spec.name}{{{items}}}"


def predicate_entails(claim_pred: PredicateSpec, obligation_pred: PredicateSpec) -> bool:
    """Syntactic entailment: same canonical hash OR AND-set containment.

    No regex subsumption, no numeric interval reasoning, no
    cross-primitive entailment. The strict rule is the soundness
    anchor — anything richer would force the plant into undecidable
    territory.
    """
    if claim_pred.name == "and" and obligation_pred.name == "and":
        claim_hashes = {
            serialize_spec(PredicateSpec(name=n, args=a))
            for n, a in claim_pred.args
        }
        for n, a in obligation_pred.args:
            child = PredicateSpec(name=n, args=a)
            if serialize_spec(child) not in claim_hashes:
                return False
        return True
    if claim_pred.name == "and" and obligation_pred.name != "and":
        target = serialize_spec(obligation_pred)
        return any(
            serialize_spec(PredicateSpec(name=n, args=a)) == target
            for n, a in claim_pred.args
        )
    if claim_pred.name != "and" and obligation_pred.name == "and":
        if len(obligation_pred.args) != 1:
            return False
        only = obligation_pred.args[0]
        return serialize_spec(PredicateSpec(name=only[0], args=only[1])) == serialize_spec(claim_pred)
    return serialize_spec(claim_pred) == serialize_spec(obligation_pred)


def is_structural(spec: PredicateSpec) -> bool:
    """A composite predicate is structural iff every conjunct is."""
    if spec.name == "and":
        return all(
            is_structural(PredicateSpec(name=n, args=a))
            for n, a in spec.args
        )
    schema = _PRIMITIVES.get(spec.name)
    return bool(schema and schema.is_structural)


def has_content_conjunct(spec: PredicateSpec) -> bool:
    """For ``and_split``: at least one conjunct must be content-bearing
    if parent is content-bearing."""
    if spec.name != "and":
        return not is_structural(spec)
    return any(
        not is_structural(PredicateSpec(name=n, args=a))
        for n, a in spec.args
    )
