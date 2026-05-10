"""Regression: ``shared_spacy`` must work from worker threads when the
pipeline was loaded on GPU in the main thread.

The bug this guards against:

* Lifespan code calls ``shared_spacy(<trf model>)`` on the main thread.
  Inside the cached factory, ``spacy.prefer_gpu()`` flips the thinc
  backend to ``CupyOps`` via ``thinc.backends.context_ops`` — a
  ``ContextVar``.
* ``ThreadPoolExecutor.submit`` (used by ``loop.run_in_executor`` in
  ``services/files.run_reingest``) does not propagate parent-thread
  contextvars to worker threads. The worker observes
  ``context_ops.get() is None``, falls back to ``NumpyOps`` via
  ``require_cpu()``, and from then on every input tensor thinc
  materialises lands on CPU while the model parameters still live on
  ``cuda:0`` — yielding ``RuntimeError: Expected all tensors to be on
  the same device``.

The fix lives in ``config.shared.shared_spacy``: each hand-out
re-asserts ``spacy.prefer_gpu()`` on the calling thread when GPU was
available at load time. This test pins that contract.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from config.shared import _spacy_cached, shared_spacy


# Skip cleanly when the host can't run GPU thinc — same skip wherever this
# test runs (CI without GPU, contributor laptop without cupy, etc.).
def _gpu_or_skip() -> str:
    try:
        import torch  # noqa: F401  — preload libcudart for cupy
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torch not importable: {e}")
    try:
        import cupy  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cupy not installed: {e}")

    import torch as _torch

    if not _torch.cuda.is_available():  # pragma: no cover
        pytest.skip("CUDA not available on this host")

    # Model resolution: prefer the project's local_storage layout (matches
    # the lifespan path), fall back to ``SPACY_TRF_MODEL`` env var so this
    # test stays runnable on hosts that keep models elsewhere.
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "local_storage" / "models" / "zh_core_web_trf"
    env_path = os.environ.get("SPACY_TRF_MODEL")
    model_path = env_path or (str(candidate) if candidate.exists() else "")
    if not model_path or not Path(model_path).exists():  # pragma: no cover
        pytest.skip(
            "no trf model on disk; set SPACY_TRF_MODEL or place "
            "local_storage/models/zh_core_web_trf"
        )
    return model_path


@pytest.fixture(scope="module")
def gpu_spacy_model() -> str:
    """Resolve a trf model path or skip the whole module."""
    return _gpu_or_skip()


@pytest.fixture(scope="module")
def main_thread_loaded_nlp(gpu_spacy_model: str):
    """Load the pipeline once on the main thread (mirrors lifespan)."""
    nlp = shared_spacy(gpu_spacy_model)
    # Sanity: main thread must always work — if this fails the test is
    # misconfigured (probably wrong model path), not a regression.
    doc = nlp("中国人民银行发布货币政策执行报告。")
    assert any(ent.text for ent in doc.ents), "main-thread sanity NER returned nothing"
    return nlp


@pytest.mark.gpu
def test_shared_spacy_inference_from_worker_thread(
    gpu_spacy_model: str, main_thread_loaded_nlp
) -> None:
    """The exact failure mode from concurrent reingest.

    Without the per-call ``prefer_gpu`` re-assertion in ``shared_spacy``
    this raises ``RuntimeError: Expected all tensors to be on the same
    device`` from inside ``torch.embedding``.
    """

    def _worker() -> list[tuple[str, str]]:
        nlp = shared_spacy(gpu_spacy_model)
        doc = nlp("中国人民银行发布货币政策执行报告。")
        return [(e.text, e.label_) for e in doc.ents]

    with ThreadPoolExecutor(max_workers=1) as ex:
        ents = ex.submit(_worker).result(timeout=60)

    assert ents, "worker-thread NER returned no entities"
    assert any("中国人民银行" in surface for surface, _ in ents)


@pytest.mark.gpu
def test_shared_spacy_concurrent_workers(
    gpu_spacy_model: str, main_thread_loaded_nlp
) -> None:
    """Multiple worker threads in parallel — mirrors the 4-file reingest fanout
    (``loop.run_in_executor`` × N concurrent reingest tasks).

    The race surface is "two workers both starting on a thread whose
    ``context_ops`` ContextVar is unset". Each must independently land
    on ``CupyOps`` and produce non-empty entities.
    """
    texts = [
        "中国人民银行发布货币政策执行报告。",
        "国家发展改革委员会公布最新规划。",
        "中国证券监督管理委员会批准上市。",
        "上海证券交易所发布交易公告。",
    ]

    def _worker(text: str) -> int:
        nlp = shared_spacy(gpu_spacy_model)
        doc = nlp(text)
        return sum(1 for _ in doc.ents)

    with ThreadPoolExecutor(max_workers=len(texts)) as ex:
        counts = list(ex.map(_worker, texts))

    assert all(c >= 1 for c in counts), f"some workers returned no entities: {counts}"


@pytest.mark.gpu
def test_shared_spacy_loaded_on_gpu(gpu_spacy_model: str) -> None:
    """Pipeline parameters must actually live on cuda — otherwise the test
    above would pass trivially (CPU-only fallback also produces entities).
    """
    nlp = shared_spacy(gpu_spacy_model)

    import torch

    def _torch_device(thinc_model) -> "torch.device | None":
        """Walk ``thinc.Model.layers`` recursively to find a PyTorchShim
        and return the device of its first parameter. Returns ``None``
        when no torch-backed shim exists in this subtree."""
        for shim in getattr(thinc_model, "shims", []) or []:
            inner = getattr(shim, "_model", None)
            if inner is None or not hasattr(inner, "parameters"):
                continue
            for p in inner.parameters():
                if isinstance(p, torch.Tensor):
                    return p.device
        for layer in getattr(thinc_model, "layers", []) or []:
            dev = _torch_device(layer)
            if dev is not None:
                return dev
        return None

    cuda_param_seen = False
    for _, comp in nlp.pipeline:
        model = getattr(comp, "model", None)
        if model is None:
            continue
        dev = _torch_device(model)
        if dev is not None and dev.type == "cuda":
            cuda_param_seen = True
            break

    assert cuda_param_seen, (
        "no CUDA parameters found in pipeline — GPU path didn't activate; "
        "the threaded test would have passed for the wrong reason"
    )


@pytest.mark.gpu
def test_cache_clear_then_threaded_reload(gpu_spacy_model: str) -> None:
    """Cold-start path: even when the cache is dropped first, a worker
    thread asking for the pipeline must end up with both the load and
    its own ops backend on GPU.

    This covers the ``clear_caches()`` admin path followed by a worker
    thread being the very first caller after the wipe.
    """
    _spacy_cached.cache_clear()

    def _worker() -> list[tuple[str, str]]:
        nlp = shared_spacy(gpu_spacy_model)
        doc = nlp("中国人民银行发布货币政策执行报告。")
        return [(e.text, e.label_) for e in doc.ents]

    with ThreadPoolExecutor(max_workers=1) as ex:
        ents = ex.submit(_worker).result(timeout=120)

    assert ents
