# Code guide: singing voice timbre attribute prediction

[Back to the project README](../README.md)

This directory contains a 44-dimensional acoustic feature extractor and a
FACodec-based source/target-domain timbre regression workflow. The annotation JSON is available at `../SDVED_TDA.json`. No dataset or
local experiment path is built into the scripts; users provide
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

## Scope

This guide documents the currently uploaded scripts. There is no standalone inference script or released timbre prediction checkpoint. Commands below are based on source inspection; a complete training run has not been verified as part of this documentation update.

## Installation

Run the commands below from `Codes/` (`cd Codes` from the repository root).

Python 3.10 or newer is recommended by the existing code documentation; an exact tested dependency environment has not been pinned. Install a matching PyTorch/torchaudio
build for your CPU or CUDA environment first, then install the remaining
dependencies:

```bash
pip install -r requirements.txt
git clone https://github.com/open-mmlab/Amphion.git
```

By default, the training scripts look for Amphion in `Codes/Amphion`, relative to the scripts. If it is
stored elsewhere, set its repository path explicitly:

```bash
export AMPHION_ROOT=/path/to/Amphion
```

On Windows PowerShell, use `$env:AMPHION_ROOT = "C:/path/to/Amphion"`. The scripts import the `Amphion` package by name, so retain that checkout directory name. Multiline commands below use Bash continuation syntax; on PowerShell, put each command on one line.

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

The scripts read CSV or Excel tables, not the annotation JSON directly. For SDVED-TDA, merge the JSON with the feature CSV by a verified sample identifier, derive `speaker_id` from the audio path, and select one descriptor as the target. Ensure file stems are unique under the audio root: the current matching logic uses stems, not full relative paths.

You may retain a descriptor column such as `bright` and pass `--score-column bright`, rather than renaming it to `score`. The 18-descriptor release does not mean that a single training run predicts 18 outputs. Prepare each descriptor and any male/female subset separately. Source-domain opposite-descriptor mappings also need to be prepared explicitly; the trainers do not apply them automatically.

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

For a from-scratch run, omit `--pretrained` **and set `--freeze-level 0`** so randomly initialized head layers remain trainable. Useful options
include `--freeze-level {0,1,2}`, `--val-ratio 0.2`, and `--num-workers 4`.

Both training scripts save `best_hybrid_model.pth` and `final_hybrid_model.pth` in the selected checkpoint directory. Use a separate directory for each experiment to avoid overwriting another model. These files contain the prediction head state and limited metadata; they do not bundle FACodec or feature normalization statistics.

## Bradley-Terry score generation

Run this utility separately for each descriptor and intended comparison group. The parser ignores the text before `:` and pools all pairs from the input file; mixing attributes in one file would fit one combined score vector.

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

## Current implementation and reproduction notes

| Setting | Source script | Target script |
| --- | --- | --- |
| Split | Sample-level 80/20, split seed 42 | Singer-level, `--val-ratio` default 0.2, split seed 42 |
| Learning rate | `1e-3` | `1e-4` |
| Maximum epochs | 50 | 30 |
| Batch size | 128 | 128 |
| Training loss | MSE + 0.5 × MAE | MSE + 0.5 × MAE |
| Feature Z-score statistics | Fitted on the training subset | Currently computed from the full valid label table before splitting |
| Audio duration | Two-second crops | Deterministic two-second segments after optional edge trimming |

Learning rate, epoch count, and batch size are currently configured in each script's `Config` class; they are not command-line flags. The split seed does not seed every source of training randomness.

`--freeze-level 0` trains all head parameters; level 1 freezes the embedding normalization and manual-feature branch; level 2 additionally freezes the first fusion block. FACodec remains frozen at all levels.
in our work, we used level 2 and modify the `frozen_indices` (434 line) to control the freeze layers & the coefficient at #376 line to control the learning rate.

The manuscript reports a source learning rate of `1e-4` and equal MAE/MSE weighting. The public defaults above differ, and the source split is not the official dataset split described in the paper. Do not interpret default runs as exact reproductions of the paper's tables.

The target script selects its checkpoint using validation loss and then reports MAE on those same validation segments. Longer files contribute more segments. This is not a separate test-set evaluation or a file-level aggregation of predictions.

Target feature normalization currently uses validation information. For evaluation without that leakage, the implementation needs to fit statistics after splitting, on training data only. Both scripts also need to persist the fitted statistics before the resulting checkpoints can support consistent standalone inference. These are implementation limitations; this README does not change the code.

`BradleyTerry.py` currently calls `choix.ilsr_pairwise` and clips the Z-score mapping to [0, 10]. Its `alpha` option should not by itself be taken as proof that this solver implements the manuscript's stated Gaussian-prior PMLE objective. Verify that correspondence before claiming exact label-generation reproduction.

Exact paper split manifests, an independent test evaluation entry point, and a standalone inference command are not currently included.
