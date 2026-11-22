"""Attaques de robustesse, appliquées au signal normalisé à l'ancrage.

vitesse       rééchantillonnage : hauteur ET durée changent (comme dans Dugelay et al.)
mp3           aller-retour MP3 (LAME via ffmpeg), à 44,1 kHz par défaut
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .align import align_to
from .io import decode_mono, run_ffmpeg, write_wav
from .resample import resample


def speed(x: np.ndarray, sr: int, factor: float, backend: str = "soxr") -> np.ndarray:
    """Lecture `factor` fois plus rapide : durée / factor, fréquences x factor."""
    if factor <= 0:
        raise ValueError("facteur de vitesse > 0 requis")
    return resample(x, sr * factor, sr, backend)


def mp3_roundtrip(x: np.ndarray, sr: int, kbps: int, mp3_sr: int = 44100,
                  backend: str = "soxr") -> np.ndarray:
    """x -> mp3_sr -> MP3 (LAME, `kbps`, mono) -> décodage -> sr ; retard d'encodeur compensé.

    À 16 kHz l'encodage direct n'autorise pas 192 ou 320 kbit/s (MPEG-2 couche III) : on passe donc
    par `mp3_sr` (44,1 kHz par défaut), comme une piste réellement distribuée.
    """
    up = resample(x, sr, mp3_sr, backend)
    with tempfile.TemporaryDirectory(prefix="md_mp3_") as d:
        src, mp3 = Path(d) / "in.wav", Path(d) / "out.mp3"
        write_wav(src, up, mp3_sr, dtype="float32")
        run_ffmpeg(["-i", str(src), "-ac", "1", "-c:a", "libmp3lame", "-b:a", f"{int(kbps)}k", str(mp3)])
        dec, sr_dec = decode_mono(mp3)
    down = resample(dec, sr_dec, sr, backend)
    down, _ = align_to(x, down, max_lag=int(0.05 * sr))
    return down


def attack_name(spec: dict[str, Any]) -> str:
    kind = spec["kind"]
    if kind == "speed":
        return f"speed_{spec['factor']:g}"
    if kind == "mp3":
        return f"mp3_{int(spec['kbps'])}"
    if kind == "none":
        return "clean"
    raise ValueError(f"attaque inconnue : {kind}")


def apply_attack(x: np.ndarray, sr: int, spec: dict[str, Any], backend: str = "soxr") -> np.ndarray:
    kind = spec["kind"]
    if kind == "none":
        return x
    if kind == "speed":
        return speed(x, sr, spec["factor"], backend)
    if kind == "mp3":
        return mp3_roundtrip(x, sr, spec["kbps"], spec.get("mp3_sr", 44100), backend)
    raise ValueError(f"attaque inconnue : {kind}")


class AttackTransform:
    """Transformation appelable par SegmentSet : (x, sr, index) -> x attaqué."""

    def __init__(self, spec: dict[str, Any], backend: str = "soxr"):
        self.spec, self.backend = spec, backend
        self.name = attack_name(spec)

    def __call__(self, x: np.ndarray, sr: int, index: int) -> np.ndarray:
        return apply_attack(x, sr, self.spec, self.backend)


def default_attack_grid(cfg_attacks: dict[str, Any]) -> list[dict[str, Any]]:
    """Grille d'attaques à partir de `eval.attacks` de la configuration."""
    grid: list[dict[str, Any]] = [{"kind": "speed", "factor": float(f)} for f in cfg_attacks.get("speed", [])]
    grid += [{"kind": "mp3", "kbps": int(k)} for k in cfg_attacks.get("mp3_kbps", [])]
    return grid
