44-D Timbre Regression
======================

This repository contains a 44-dimensional acoustic feature extractor and a
FACodec-based source/target-domain timbre regression workflow. No dataset,
label file, or local experiment path is built into the code; users provide
their own paths through command-line arguments.

## 44-dimensional feature definition

The public feature set is fixed to the following schema:

- 5 acoustic summaries: `centroid_mean`, `flux_mean_low`, `hnr_praat`,
  `f1_praat`, and `f2_praat`
- 13 MFCC means: `mfcc_1_mean` through `mfcc_13_mean`
- 13 first-order delta means: `mfcc_delta_1_mean` through
  `mfcc_delta_13_mean`
- 13 second-order delta means: `mfcc_delta2_1_mean` through
  `mfcc_delta2_13_mean`

The two experimental features `cpp_praat` and `inharmonicity_mean` are not
part of this release. Both training scripts select the schema above by column
name and reject CSV files that are missing any of the 44 columns.

## Installation

Python 3.10 or newer is recommended. Install a matching PyTorch/torchaudio
build for your CPU or CUDA environment first, then install the remaining
dependencies:

```bash
pip install -r requirements.txt
git clone https://github.com/open-mmlab/Amphion.git
```

By default, the training scripts look for Amphion in `./Amphion`. If it is
stored elsewhere, set its repository path explicitly:

```bash
export AMPHION_ROOT=/path/to/Amphion
```

FACodec weights are downloaded from Hugging Face on the first training run.

## 1. Generate the 44-D feature CSV

WAV files are found recursively. `filename` in the output CSV is the path
relative to the supplied audio folder.

```bash
python feature_extract_filewise.py \
  --folder /path/to/audio \
  --output outputs/features_44d.csv \
  --workers 4
```

Use `--max-files 10` for a small test run. Run with `--workers 1` when
debugging.

## 2. Prepare the training CSV

Merge your labels into the generated feature CSV. The default metadata
columns are:

```text
filename,speaker_id,score,<the 44 feature columns>
```

- `filename`: audio filename or relative path; matching is performed by file
  stem.
- `speaker_id`: speaker identity used by the target-domain speaker split.
- `score`: regression target.

Different metadata names are supported with `--filename-column`,
`--speaker-column`, and `--score-column`. Extra CSV columns are ignored.

## 3. Train the source-domain model

```bash
python train_source.py \
  --labels /path/to/source_labels_and_features.csv \
  --audio-root /path/to/source_audio \
  --checkpoint-dir outputs/source_checkpoints
```

The best transferable checkpoint is written to
`outputs/source_checkpoints/best_hybrid_model.pth`.

## 4. Fine-tune on a target domain

```bash
python train_target.py \
  --labels /path/to/target_labels_and_features.csv \
  --audio-root /path/to/target_audio \
  --pretrained outputs/source_checkpoints/best_hybrid_model.pth \
  --checkpoint-dir outputs/target_checkpoints
```

Omit `--pretrained` to train the target model from scratch. Useful options
include `--freeze-level {0,1,2}`, `--val-ratio 0.2`, and `--num-workers 4`.

## Bradley-Terry score generation

Each non-comment input line contains an arbitrary attribute label, followed
by alternating winner/loser item names:

```text
brightness: speaker_01 speaker_02 speaker_03 speaker_01
brightness: speaker_02 speaker_03
```

Generate a CSV containing raw and 0-to-10 mapped scores with:

```bash
python BradleyTerry.py \
  --input /path/to/pairwise_comparisons.txt \
  --output outputs/bradley_terry_scores.csv
```

Use `python <script>.py --help` to see every available option.
