"""Inventory adapter — read-only domain enumeration.

The kernel does not own document storage. It asks the inventory which
units exist for a given (ScopeRef, unit_type), and that answer defines
the domain over which ``complete_scan`` checks coverage. Soundness is
relative to whatever this adapter returns.
"""

from typing import Optional, Protocol, runtime_checkable

from agentic.closure.obligation import ScopeRef, UnitType


class UnknownScopeError(ValueError):
    """ScopeRef points at files / sections the underlying store has no record of."""


@runtime_checkable
class Inventory(Protocol):
    def units(self, scope: ScopeRef, unit_type: UnitType) -> frozenset[str]: ...


class InventoryAdapter:
    """Wraps ``storage.inventory_store.InventoryStore`` for the kernel.

    Caches results per (canonical_scope_id, unit_type) so a finalize
    cycle does not redo enumeration on every closure rule. Validation
    can be disabled in tests via ``validate=False``.
    """

    def __init__(self, store, *, validate: bool = True) -> None:
        self._store = store
        self._validate = validate
        self._cache: dict[tuple[str, str], frozenset[str]] = {}

    def units(self, scope: ScopeRef, unit_type: UnitType) -> frozenset[str]:
        key = (scope.canonical_scope_id, unit_type)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self._validate:
            self._check_known(scope)

        result = self._store.units(
            unit_type,
            file_ids=list(scope.file_ids) if scope.file_ids else None,
            section_ids=list(scope.section_ids) if scope.section_ids else None,
        )
        frozen = frozenset(result or ())
        self._cache[key] = frozen
        return frozen

    def _check_known(self, scope: ScopeRef) -> None:
        page_store = getattr(self._store, "page_store", None)
        if page_store is not None and scope.file_ids and hasattr(page_store, "ids"):
            known = {gid.split("/", 1)[0] for gid in page_store.ids() if "/" in gid}
            unknown = [fid for fid in scope.file_ids if fid not in known]
            if unknown:
                raise UnknownScopeError(f"unknown file_ids: {unknown}")
        if scope.section_ids and hasattr(self._store, "get"):
            for sid in scope.section_ids:
                if self._store.get(sid) is None:
                    raise UnknownScopeError(f"unknown section_id: {sid!r}")
