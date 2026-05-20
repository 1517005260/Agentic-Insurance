"""Post-hoc temperature scaling for GLiNER span confidence.

GLiNER@0.3 is uncalibrated on open-domain academic text: it emits
non-rigid spans ("this approach", "the proposed method", "X et al") at
scores 0.3–0.6 that are flagged as noise by a reference annotator.
Temperature scaling maps the raw score s → s/T and re-applies the
threshold. T > 1 tightens the effective threshold (reducing
over-generation); T < 1 loosens it (recall-oriented).

The temperature T is fitted offline on a silver span dev set
(``experiments/ner_calibration.py``). The dev set is LLM-annotated
(gpt-4o-mini via the project's chat API), collected from 40 sampled
sentences of the 154-doc stock, and is **dev-only / not the evaluation
set**. The annotator bias is declared in the dev-set JSON (annotator_model
field). Calibration is a standard post-hoc method independent of
domain: the same temperature can be re-fitted on any annotated sample
without touching the model or the label list.

References:
  Platt (1999) "Probabilistic Outputs for SVMs"; used in NER context as
  temperature scaling (single-parameter special case of Platt scaling).
  Guo et al. (2017) "On Calibration of Modern Neural Networks" (ICML).
"""
import math
from typing import List, Tuple


def expected_calibration_error(
    scores: List[float],
    labels: List[int],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) for binary classification.

    Bins spans by predicted score, computes |mean_score - fraction_positive|
    per bin, and returns the weighted average.

    Args:
        scores: Raw or calibrated GLiNER confidence scores in [0, 1].
        labels: Binary ground-truth (1 = entity, 0 = noise).
        n_bins: Number of equal-width bins across [0, 1].

    Returns:
        ECE in [0, 1]; 0.0 = perfectly calibrated.
    """
    if not scores:
        return 0.0
    bin_size = 1.0 / n_bins
    total = len(scores)
    ece = 0.0
    for b in range(n_bins):
        lo = b * bin_size
        hi = lo + bin_size
        idxs = [i for i, s in enumerate(scores) if lo <= s < hi]
        if not idxs:
            continue
        n_b = len(idxs)
        mean_score = sum(scores[i] for i in idxs) / n_b
        frac_pos = sum(labels[i] for i in idxs) / n_b
        ece += (n_b / total) * abs(mean_score - frac_pos)
    return ece


def reliability_diagram_buckets(
    scores: List[float],
    labels: List[int],
    n_bins: int = 10,
) -> List[dict]:
    """Return per-bin stats for a reliability diagram.

    Each dict has: ``lo``, ``hi``, ``n``, ``mean_score``, ``frac_pos``.
    Empty bins are omitted.
    """
    bin_size = 1.0 / n_bins
    buckets = []
    for b in range(n_bins):
        lo = b * bin_size
        hi = lo + bin_size
        idxs = [i for i, s in enumerate(scores) if lo <= s < hi]
        if not idxs:
            continue
        n_b = len(idxs)
        buckets.append({
            "lo": round(lo, 3),
            "hi": round(hi, 3),
            "n": n_b,
            "mean_score": round(sum(scores[i] for i in idxs) / n_b, 4),
            "frac_pos": round(sum(labels[i] for i in idxs) / n_b, 4),
        })
    return buckets


def fit_temperature(
    scores: List[float],
    labels: List[int],
    n_iters: int = 200,
    tol: float = 1e-6,
) -> float:
    """Fit a single temperature parameter T on a silver dev set.

    Minimises cross-entropy loss L(T) = -sum( y*log(s/T) + (1-y)*log(1-s/T) )
    using scipy.optimize.minimize_scalar over T in [0.1, 10.0].

    Uses scipy (already installed in the server venv). Does not require
    netcal, sklearn, or any additional dependency.

    Args:
        scores: Raw GLiNER scores in (0, 1).
        labels: Binary (1 = entity, 0 = noise).

    Returns:
        Optimal T; T=1.0 if the optimisation fails or the sample is empty.
    """
    if not scores:
        return 1.0

    def nll(T: float) -> float:
        eps = 1e-9
        total = 0.0
        for s, y in zip(scores, labels):
            p = min(max(s / T, eps), 1.0 - eps)
            total -= y * math.log(p) + (1 - y) * math.log(1 - p)
        return total

    try:
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded",
                                 options={"maxiter": n_iters, "xatol": tol})
        return float(result.x)
    except Exception:
        return 1.0


def apply_temperature(score: float, temperature: float) -> float:
    """Scale a raw GLiNER score by the calibration temperature.

    The scaled score ``score / temperature`` is compared against the
    original threshold (``gliner_threshold``). T > 1 raises the
    effective threshold (tighter); T < 1 lowers it (looser).
    """
    if temperature <= 0.0:
        return score
    return score / temperature
