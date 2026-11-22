"""Calibration : erreur de calibration attendue (ECE)."""

from __future__ import annotations

import numpy as np


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error de la probabilité « faux » (bins d'égale largeur)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = which == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)
