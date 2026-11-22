"""Codecs neuronaux via `transformers` (extra « torch ») : EnCodec 24 kHz et EnCodec 32 kHz de MusicGen.

Un seul paquet et des poids sur le Hugging Face Hub. Tout le code de ce module est exécuté en
`torch.inference_mode()`.
"""

from __future__ import annotations

import numpy as np

from .base import Codec


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch est requis pour les codecs neuronaux : `uv sync --extra torch`") from exc
    return torch


def _pick_device(device: str | None) -> str:
    torch = _torch()
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class _HFCodec(Codec):
    """Squelette : charge le modèle une fois, reconstruit, recoupe à la longueur d'entrée."""

    def __init__(self, hf_id: str, name: str, device: str | None = None):
        self.hf_id, self.name, self.device = hf_id, name, device
        self._model = None

    # --- à définir par les sous-classes
    def _load(self):  # -> modèle HF en mode eval, et met à jour self.native_sr
        raise NotImplementedError

    def _forward(self, x_t):  # (1, 1, T) -> tenseur audio reconstruit
        raise NotImplementedError

    def _ensure(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def prepare(self) -> None:
        self._ensure()

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        torch = _torch()
        self._ensure()
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        x_t = torch.from_numpy(x)[None, None, :].to(self._dev)
        with torch.inference_mode():
            y = self._forward(x_t)
        y = y.detach().float().reshape(-1).cpu().numpy()
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        return y[:n].astype(np.float32)


class EncodecCodec(_HFCodec):
    """EnCodec 24 kHz mono (facebook/encodec_24khz) ; `bandwidth` en kbit/s parmi 1,5 / 3 / 6 / 12 / 24."""

    def __init__(self, hf_id: str = "facebook/encodec_24khz", bandwidth: float = 6.0,
                 name: str = "encodec", device: str | None = None):
        super().__init__(hf_id, name, device)
        self.bandwidth = float(bandwidth)
        self.native_sr = 24000

    def _load(self):
        from transformers import EncodecModel
        self._dev = _pick_device(self.device)
        model = EncodecModel.from_pretrained(self.hf_id).to(self._dev).eval()
        allowed = list(model.config.target_bandwidths)
        if self.bandwidth not in allowed:
            raise ValueError(f"bandwidth {self.bandwidth} hors de {allowed} pour {self.hf_id}")
        self.native_sr = int(model.config.sampling_rate)
        return model

    def _forward(self, x_t):
        return self._model(x_t, bandwidth=self.bandwidth).audio_values


class MusicgenEncodecCodec(_HFCodec):
    """EnCodec 32 kHz embarqué dans MusicGen (50 trames/s, 2,2 kbit/s : `target_bandwidths` du checkpoint)."""

    def __init__(self, hf_id: str = "facebook/musicgen-small", name: str = "encodec_32k_musicgen",
                 device: str | None = None):
        super().__init__(hf_id, name, device)
        self.native_sr = 32000

    def _load(self):
        from transformers import MusicgenForConditionalGeneration
        self._dev = _pick_device(self.device)
        full = MusicgenForConditionalGeneration.from_pretrained(self.hf_id)
        enc = full.audio_encoder.to(self._dev).eval()
        del full
        self.native_sr = int(enc.config.sampling_rate)
        return enc

    def _forward(self, x_t):
        return self._model(x_t).audio_values
