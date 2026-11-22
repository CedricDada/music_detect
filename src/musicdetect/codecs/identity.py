"""Codec identité : contrôle négatif de la chaîne (réel et faux ne diffèrent que par le traitement commun)."""

from __future__ import annotations

import numpy as np

from .base import Codec


class IdentityCodec(Codec):
    def __init__(self, native_sr: int = 24000, name: str = "identity"):
        self.native_sr, self.name = int(native_sr), name

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32).copy()
