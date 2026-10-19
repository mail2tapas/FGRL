"""Statistics for Section 5.7: Friedman test, Nemenyi post hoc test, Cohen's d.

Input is a matrix scores[block, method], one block per (workload, seed)
combination, all methods evaluated on the same blocks.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import numpy as np
from scipy import stats


def friedman(scores: np.ndarray) -> Dict[str, float]:
    res = stats.friedmanchisquare(*[scores[:, j] for j in range(scores.shape[1])])
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


def average_ranks(scores: np.ndarray, lower_is_better: bool = True) -> np.ndarray:
    s = scores if lower_is_better else -scores
    return np.mean(np.apply_along_axis(stats.rankdata, 1, s), axis=0)


def nemenyi(scores: np.ndarray, names: List[str], lower_is_better: bool = True) -> List[Dict]:
    """Pairwise Nemenyi p-values from the studentized range distribution."""
    n, k = scores.shape
    R = average_ranks(scores, lower_is_better)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    out = []
    for i, j in combinations(range(k), 2):
        q = abs(R[i] - R[j]) / se * np.sqrt(2)
        p = float(stats.studentized_range.sf(q, k, np.inf))
        out.append({"a": names[i], "b": names[j], "rank_a": float(R[i]), "rank_b": float(R[j]), "p_value": p})
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD effect size (mean(a) - mean(b)) / s_pooled."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("inf")


def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Paired effect size for matched blocks: mean(a - b) / sd(a - b).
    Preferred when every method is evaluated on the same (seed, workflow)
    blocks, because it removes the between-workload variation."""
    d = np.asarray(a, float) - np.asarray(b, float)
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else float("inf")
