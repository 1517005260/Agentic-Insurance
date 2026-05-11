"""Reranker client — local Qwen3-Reranker (instruction-tuned cross-encoder).

The model is a causal LM with a yes/no scoring head: each (query, doc)
pair is scored independently as ``softmax([logit(no), logit(yes)])[yes]``.
This is true *pairwise* — no cross-request normalization, so the score
of pair A is comparable to the score of pair B from a different call.
That property is what makes the score usable as a threshold for the
alias-edge veto layer (see ``src/ingestion/index/linear_rag/disambig.py``).

The instruction-tuned framing lets the caller pin the label space:
``"Retrieve passages relevant to the query"`` for RAG page reranking,
``"Are these the same real-world entity?"`` for ER alias verification.
The instruction is the **only** lever that lets one model serve both
use cases without fine-tuning.

Two public entry points:

* :meth:`RerankClient.rerank` — query + N documents → top-N sorted by
  score; used by ``src/rag/rerank.py`` for page reranking.
* :meth:`RerankClient.score_pairs` — list of (query, doc) tuples →
  list of scores, no sorting; the ER use case where every pair has a
  different query.

GPU FP16 is mandatory. Weights live at
``settings.rerank_model_dir()`` (``STORAGE_PATH/models/Qwen3-Reranker-0.6B``),
pre-fetched by ``python download_models.py``.
"""

from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple

import torch

from config.settings import RERANK_MODEL_ID, rerank_model_dir
from config.shared import shared_qwen_reranker


# Default instruction — biased toward the *retrieval* label space
# because ``rerank(query, documents)`` is a passage-relevance task.
# The ER alias path passes its own ER-specific instruction at call
# time (see ``disambig.reranker_veto``).
DEFAULT_INSTRUCTION = (
    "Given a search query, retrieve passages that are relevant to the query."
)

# Qwen3-Reranker chat template — model-card-prescribed wrappers around
# the (instruction, query, document) triple. The prefix/suffix tokens
# are computed once per (tokenizer, instruction) combo and concatenated
# around the body tokens at score time. Keep these in lockstep with the
# Qwen3-Reranker model card if you upgrade the checkpoint.
_PREFIX_TEMPLATE = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class RerankResult(dict):
    """{"index": int, "relevance_score": float}; dict for cheap json log."""


class RerankClient:
    """Pairwise cross-encoder reranker backed by Qwen3-Reranker-0.6B.

    Thread-safe: the cached (tokenizer, model) handle is shared across
    threads under PyTorch's read-only forward; per-instance state is
    immutable after ``__init__``.
    """

    def __init__(
        self,
        model_dir: Optional[Any] = None,
        max_length: int = 1024,
        batch_size: int = 8,
    ):
        # ``model_dir`` overrides the default location (rarely useful
        # outside tests); production callers leave it None so the
        # standard ``STORAGE_PATH/models/...`` resolution applies.
        resolved_dir = model_dir if model_dir is not None else rerank_model_dir()
        self.model_id = RERANK_MODEL_ID
        self.model_dir = resolved_dir
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        # Tokenizer + model + cached yes/no token ids — see shared.py.
        self._tokenizer, self._model, self._yes_id, self._no_id = (
            shared_qwen_reranker(resolved_dir)
        )
        self._prefix_tokens = self._tokenizer.encode(_PREFIX_TEMPLATE, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(_SUFFIX, add_special_tokens=False)

    def available(self) -> bool:
        # Always True once construction succeeds — failures raise in
        # ``__init__``. Kept as a method so callers can probe with a
        # uniform interface (test doubles can return False).
        return True

    @torch.no_grad()
    def score_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
        instruction: Optional[str] = None,
    ) -> List[float]:
        """Score each (query, document) pair independently.

        Returns a list of floats in ``[0, 1]`` parallel to ``pairs``;
        ``>0.5`` means the model favors "yes" (= relevant / same entity
        depending on the instruction). Pairs are batched in
        ``self.batch_size`` chunks for GPU throughput; no truncation
        beyond ``self.max_length`` minus the prefix/suffix budget.
        """
        if not pairs:
            return []
        instruct = instruction or DEFAULT_INSTRUCTION
        scores: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            bodies = [
                f"<Instruct>: {instruct}\n<Query>: {q}\n<Document>: {d}"
                for q, d in chunk
            ]
            inputs = self._tokenizer(
                bodies,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens),
            )
            # Wrap each body with the chat-template prefix/suffix —
            # see Qwen3-Reranker model card. We add them at the token
            # level (not the string level) so a long body never bumps
            # the prefix tokens out under truncation.
            for i in range(len(inputs["input_ids"])):
                inputs["input_ids"][i] = (
                    self._prefix_tokens + inputs["input_ids"][i] + self._suffix_tokens
                )
            inputs = self._tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to("cuda")
            logits = self._model(**inputs).logits[:, -1, :]
            yes_logits = logits[:, self._yes_id]
            no_logits = logits[:, self._no_id]
            stacked = torch.stack([no_logits, yes_logits], dim=1)
            log_prob = torch.nn.functional.log_softmax(stacked, dim=1)
            scores.extend(log_prob[:, 1].exp().tolist())
        return scores

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        """Score every (query, doc) pair, return the top ``top_n`` sorted desc.

        ``index`` is the position in ``documents``; ``relevance_score``
        is the yes-probability. Identical pairs always produce identical
        scores — no cross-request normalization, so scores are
        comparable across calls.
        """
        if not documents:
            return []
        pairs = [(query, d) for d in documents]
        raw_scores = self.score_pairs(pairs, instruction=instruction)
        ranked = sorted(
            enumerate(raw_scores), key=lambda kv: kv[1], reverse=True
        )
        cap = min(int(top_n), len(documents))
        return [
            RerankResult(index=idx, relevance_score=float(score))
            for idx, score in ranked[:cap]
        ]


@lru_cache(maxsize=1)
def get_cached_rerank_client() -> "RerankClient":
    """Process-wide singleton for the no-arg :class:`RerankClient`.

    Construction is cheap (the heavy model load is memoised in
    ``shared_qwen_reranker``); this layer memoises the thin wrapper so
    callers using the ``client or get_cached(...)`` pattern don't
    re-run the constructor on every request.
    """
    return RerankClient()
