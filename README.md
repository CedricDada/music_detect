# musicdetect

Réplication de trois travaux de Deezer sur la détection de musique générée par IA, dans un cadre où aucune
musique générée ne sert à l'entraînement. Les détecteurs apprennent à distinguer 2 000 pistes de FMA de leur
reconstruction par EnCodec, puis sont testés sur les cinq générateurs de FakeMusicCaps (MusicGen, Stable Audio
Open, AudioLDM2, MusicLDM, Mustango). Une chaîne commune est conçue pour qu'un réel et son faux ne diffèrent que
par le codec : même aller-retour de rééchantillonnage (24 puis 16 kHz), alignement, découpe aux mêmes instants,
niveau RMS à −23 dBFS. Le rapport complet est dans [`report/main.pdf`](report/main.pdf).

![Fakeprint d'une reconstruction EnCodec](assets/readme/fakeprint.gif)

Ce que cherche le détecteur M1 : en moyennant 40 fenêtres du spectre d'une reconstruction EnCodec, la musique se
dilue et les pics laissés par le décodeur ressortent, espacés de 75 Hz, la cadence d'EnCodec 24 kHz (24 000
échantillons par seconde, 320 par trame). Ce profil, moins son enveloppe, est le fakeprint.

Travaux répliqués : Afchar et al. 2024 et ICASSP 2025 ([2405.04181](https://arxiv.org/abs/2405.04181),
[2501.10111](https://arxiv.org/abs/2501.10111)), Afchar et al., ISMIR 2025
([2506.19108](https://arxiv.org/abs/2506.19108)), Dugelay et al., ISMIR 2026
([2607.27454](https://arxiv.org/abs/2607.27454)).

## Résultats

| Question | Verdict | Chiffres |
|---|---|---|
| Un détecteur entraîné sur des reconstructions détecte-t-il de vrais générateurs ? | impossible à conclure ici | AUC de 0,62 à 0,94 pour le fakeprint (M1), mais 0,61 à 0,97 pour un détecteur trivial (M0) |
| Les artefacts suivent-ils la cadence du décodeur ? | soutenu, avec une réserve | entraîné sur l'EnCodec de MusicGen (50 trames/s), M1 passe de 0,796 à 0,997 sur MusicGen ; mais ce codec a aussi un débit plus faible (2,2 contre 6 kbit/s) |
| Une partie de la performance vient-elle du format des fichiers ? | oui, limite du protocole | M0 atteint 0,747 sur MusicGen |
| Le fakeprint en log-fréquence résiste-t-il à la vitesse ? | répliqué, et étendu au MP3 | vitesse ×1,1 : M1 de 0,796 à 0,499, M3 sans perte |

![Contraste de peigne des poids de M1](assets/readme/peigne_appris.gif)

Le contraste de peigne des poids de M1 (leur valeur moyenne, en écarts-types, aux multiples d'un espacement
donné) culmine à 75 et 150 Hz : M1 a appris le peigne de la cadence de son codec.

![Peigne de MusicGen à 50 Hz, poids de M1 puis de l'oracle](assets/readme/oracle_50hz.gif)

MusicGen laisse un peigne espacé de 50 Hz, dont M1 ne voit qu'un pic sur trois. Entraîné sur l'EnCodec de
MusicGen, l'oracle s'aligne sur 50 Hz et passe de 0,796 à 0,997 sur MusicGen ; le débit plus faible de ce codec
reste un facteur confondu.

![Accélérer de 10 % : les pics quittent les poids de M1](assets/readme/vitesse.gif)

Schéma : accélérer de 10 % multiplie toutes les fréquences par 1,1, les pics quittent les poids de M1 et son AUC
sur MusicGen tombe à 0,499. Sur une échelle logarithmique, ce décalage devient une translation que M3 rattrape
(0,761 sans attaque, 0,781 sous l'accélération).

Contrôles : sur des artistes jamais vus, M1 sépare parfaitement FMA et FMA-EnCodec (AUC 1,000). Avec un codec
identité, dont les faux sont identiques aux réels, tous les détecteurs obtiennent 0,500 : rien hors du signal ne
trahit la classe.

Limite principale : FakeMusicCaps ne contient pas les vrais extraits MusicCaps, donc les négatifs des tests sont
des morceaux FMA. Les autres limites sont dans le rapport.

## Installation

Il faut uv, Python 3.13 et ffmpeg ≥ 6 avec `libmp3lame`.

```bash
uv sync                  # suffit pour l'analyse
uv sync --extra torch    # pour tout recalculer (codecs neuronaux, M2, M3) ; --extra all ajoute M4 (SONICS)
```

M0 et M1 sont enregistrés par pickle : ne pas changer la version de scikit-learn fixée par `uv.lock` (1.9.1).

## Reproduire les chiffres du rapport (quelques minutes)

Le dépôt contient les scores par segment de toutes les évaluations (`results/phase1*/`), les index de segments
et le manifeste FMA (`data/work/`), ainsi que les modèles entraînés (`data/work/models/`). L'analyse recalcule à
partir de ces fichiers les AUC, intervalles, comparaisons appariées et figures du rapport :

```bash
uv run python -m musicdetect.analysis --results results \
    --fma-index data/work/index_fma_phase1.csv --fmc-index data/work/index_fmc_phase1.csv \
    --manifest data/work/manifest_fma.csv --out results/analysis
cd report && latexmk -pdf main.tex
```

Les comparaisons principales sont dans `results/analysis/journal.txt`, les figures dans
`results/analysis/figures/`. Le contraste de peigne demande l'audio : ses sorties sont fournies telles quelles
dans `results/analysis/peigne/` et `peigne_oracle/`.

## Tout recalculer depuis l'audio

Données à placer dans `data/raw/` :

- FMA : `fma_metadata.zip` (342 Mio) et `fma_small.zip` (7,2 Gio), liens dans le README de
  [mdeff/fma](https://github.com/mdeff/fma), à décompresser dans `data/raw/fma/`.
- FakeMusicCaps : sur Zenodo ([13732524](https://zenodo.org/records/13732524)), où l'accès aux fichiers se
  demande depuis la page du jeu de données. À décompresser dans `data/raw/FakeMusicCaps/`.

Puis, dans l'ordre (bash ; sous Windows, remplacer `ln -s` par `mklink /J`) :

```bash
C="-c configs/phase1.yaml"
G="--test fmc:musicgen --test fmc:stable_audio_open --test fmc:audioldm2 --test fmc:musicldm --test fmc:mustango"
tests() { echo "--test fma:encodec_24k_6kbps --test fma:identity_24k $([ $1 = encodec_24k_12kbps ] && echo --test fma:$1) $G"; }

# 1. Segments : 2 000 pistes FMA, faux EnCodec et codec identité, clips FakeMusicCaps
uv run musicdetect manifest $C --tag phase1
uv run musicdetect build-fma $C --tag phase1 --codec encodec_24k_6kbps --codec encodec_24k_12kbps --codec identity_24k
uv run musicdetect fmc-index $C --tag phase1 && uv run musicdetect build-fmc $C --tag phase1

# 2. M0, M1 (déterministes) et M4 (pré-entraîné)
for c in encodec_24k_6kbps encodec_24k_12kbps; do for d in m0 m1; do
  uv run musicdetect train $C --tag phase1 --detector $d --codec $c
  uv run musicdetect evaluate $C --tag phase1 --detector $d --codec $c $(tests $c)
done; done
uv run musicdetect evaluate $C --tag phase1 --detector m4 --codec encodec_24k_6kbps $(tests encodec_24k_6kbps)

# 3. M2 et M3 : une graine par tag, sur les mêmes segments
for s in 1 2 3; do
  for k in fma fmc; do
    cp data/work/index_${k}_phase1.csv data/work/index_${k}_phase1_s$s.csv
    ln -s "$PWD/data/work/segments/${k}_phase1" "data/work/segments/${k}_phase1_s$s"
  done
  for c in encodec_24k_6kbps encodec_24k_12kbps; do for d in m2 m3; do
    uv run musicdetect train $C --tag phase1_s$s --detector $d --codec $c --set $d.seed=$s --jobs 4
    uv run musicdetect evaluate $C --tag phase1_s$s --detector $d --codec $c $(tests $c) --jobs 4
  done; done
done

# 4. Attaques sur MusicGen (vitesse ×0,9 et ×1,1, MP3 128)
uv run musicdetect evaluate $C --tag phase1 --detector m1 --codec encodec_24k_6kbps $(tests encodec_24k_6kbps) --attacks --attack-on fmc:musicgen
for d in m3 m2; do
  uv run musicdetect evaluate $C --tag phase1_s1 --detector $d --codec encodec_24k_6kbps $(tests encodec_24k_6kbps) --attacks --attack-on fmc:musicgen --jobs 4
done

# 5. Oracle : M1 entraîné sur l'EnCodec 32 kHz de MusicGen
uv run musicdetect build-fma $C --tag phase1 --codec encodec_32k_musicgen
uv run musicdetect train $C --tag phase1 --detector m1 --codec encodec_32k_musicgen
uv run musicdetect evaluate $C --tag phase1 --detector m1 --codec encodec_32k_musicgen --test fma:encodec_32k_musicgen $(tests encodec_24k_6kbps)

# 6. Contraste de peigne (M1 24 kHz, puis oracle), analyse et copie des figures dans le rapport
P="--fma-index data/work/index_fma_phase1.csv --fma-dir data/work/segments/fma_phase1 --fmc-index data/work/index_fmc_phase1.csv --fmc-dir data/work/segments/fmc_phase1"
uv run python -m musicdetect.peigne --model data/work/models/phase1/m1__encodec_24k_6kbps.joblib $P --out results/analysis/peigne
uv run python -m musicdetect.peigne --model data/work/models/phase1/m1__encodec_32k_musicgen.joblib --frame-rate 50 $P --out results/analysis/peigne_oracle
uv run python -m musicdetect.analysis --results results --fma-index data/work/index_fma_phase1.csv \
    --fmc-index data/work/index_fmc_phase1.csv --manifest data/work/manifest_fma.csv --out results/analysis
cp results/analysis/figures/*.pdf report/figures/
cp results/analysis/peigne/fig_peigne.pdf report/figures/
cp results/analysis/peigne_oracle/fig_peigne.pdf report/figures/fig_peigne_oracle.pdf
```

Durées mesurées sur CPU : environ 6 h pour reconstruire les 2 000 pistes avec l'EnCodec de MusicGen, et pour les
attaques 70 min (M1), 2,5 h (M2) et 4,6 h (M3). Les segments occupent environ 13 Go. La graine 3 de M2 et M3 a été
entraînée sur Kaggle, dans un autre environnement logiciel : ses AUC peuvent varier légèrement d'une machine à
l'autre.

Tous les paramètres sont dans [`src/musicdetect/default.yaml`](src/musicdetect/default.yaml), surchargés par
`configs/phase1.yaml`.

## Licences

Ce code réécrit des idées de dépôts sous licence CC BY-NC 4.0 (`deezer/ismir25-ai-music-detector`,
`deezer/deepfake-detector`, `thomas0barand/robust-deepfake-detector`) ; Deezer réserve ses dérivés à la
recherche, sans usage commercial. FMA : métadonnées CC BY 4.0, audio sous licence propre à chaque artiste.
FakeMusicCaps : CC BY-NC 4.0. SONICS : données CC BY-NC, code MIT. Aucune licence n'est choisie pour ce dépôt.
Le gabarit LaTeX `sleek` (`report/packages/`) est sous licence MIT.
