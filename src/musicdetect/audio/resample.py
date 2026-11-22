"""Rééchantillonnage par soxr (qualité VHQ)."""

from __future__ import annotations

import numpy as np


def resample(x: np.ndarray, sr_in: float, sr_out: float, backend: str = "soxr",
             quality: str = "VHQ") -> np.ndarray:
    """Rééchantillonne un signal mono de `sr_in` vers `sr_out` (fréquences éventuellement non entières).

    La longueur de sortie vaut round(len(x) * sr_out / sr_in). Les appelants alignent toujours la sortie sur
    une longueur de référence.
    """
    x = np.asarray(x, dtype=np.float32)
    if float(sr_in) == float(sr_out):
        return x.copy()
    if backend != "soxr":
        raise ValueError(f"backend de rééchantillonnage inconnu : {backend}")
    import soxr
    return soxr.resample(x, sr_in, sr_out, quality=quality).astype(np.float32, copy=False)


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Coupe ou complète par des zéros (à la fin) pour obtenir exactement `n` échantillons."""
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))
