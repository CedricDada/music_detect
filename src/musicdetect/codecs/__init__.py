"""Fabrique de codecs à partir de la section `codecs` de la configuration."""

from __future__ import annotations

from .base import Codec
from .identity import IdentityCodec


def build_codec(name: str, spec: dict, device: str | None = None) -> Codec:
    """Instancie un codec d'après son `kind`. Les codecs neuronaux chargent leurs poids à la 1re utilisation."""
    kind = spec["kind"]
    if kind == "identity":
        return IdentityCodec(native_sr=spec.get("native_sr", 24000), name=name)
    from . import neural
    if kind == "encodec":
        return neural.EncodecCodec(spec["hf_id"], spec.get("bandwidth", 6.0), name, device)
    if kind == "musicgen_encodec":
        return neural.MusicgenEncodecCodec(spec["hf_id"], name, device)
    raise ValueError(f"type de codec inconnu : {kind} (codec {name})")


def build_codecs(cfg_codecs: dict, names: list[str], device: str | None = None) -> dict[str, Codec]:
    missing = [n for n in names if n not in cfg_codecs]
    if missing:
        raise KeyError(f"codecs absents de la configuration : {missing} ; disponibles : {sorted(cfg_codecs)}")
    return {n: build_codec(n, dict(cfg_codecs[n]), device) for n in names}


__all__ = ["Codec", "IdentityCodec", "build_codec", "build_codecs"]
