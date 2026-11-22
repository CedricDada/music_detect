"""musicdetect : ce que voit un détecteur de musique IA.

Sous-paquets
------------
audio       décodage, rééchantillonnage, chaîne de normalisation commune, attaques
data        FMA (musique réelle), FakeMusicCaps (test), index de segments
codecs      codecs neuronaux (EnCodec 24 kHz, EnCodec 32 kHz de MusicGen) et codec identité (contrôle négatif)
detectors   M0 (métadonnées), M1 (fakeprint), M2 (CNN), M3 (corrélation croisée), M4 (SONICS)
eval        métriques, bootstrap groupé, erreur de calibration, protocole d'évaluation
"""

__version__ = "0.1.0"
