# Reference-free Singing Voice Timbre Attribute Prediction via Perception Informed Network

# SDVED-TDA: Singing Dry Voice Evaluation Database with Timbre Descriptor Annotations

**SDVED-TDA** adds sample-level perceptual timbre annotations to the **Singing Dry Voice Evaluation Database (SDVED)**, part of the CCMusic database. While the original SDVED provides overall timbre scores, SDVED-TDA describes each singing sample using 18 timbre attributes.

This repository provides the **annotation labels** and an inter-rater reliability figure. Audio recordings, model code, and pretrained weights are not included in this release.

## Dataset overview

| Property | Description |
| --- | --- |
| Annotated samples | 132 |
| Singers | 22 |
| Samples per singer | 6 |
| Annotators | 15 trained musicians |
| Timbre descriptors | 18 |
| Rating scale | 1–10 |
| Label representation | Sample-level, annotator-averaged descriptor scores |
| File format | UTF-8 JSON |

Each sample has its own timbre attribute vector. Labels are not shared across all recordings from the same singer, allowing the annotations to preserve performance-dependent timbre variation.

## Repository contents

```text
SDVED-TDA/
├── README.md
├── SDVED_TDA.json    # Sample-level timbre attribute labels
└── ICC_pic.png       # Inter-rater reliability figure
```

The labels are available in [SDVED_TDA.json](SDVED_TDA.json).

## Annotation procedure

Fifteen trained musicians (eight males and seven females, aged 20–30) participated in the listening study. Their average musical experience was 15.9 years.

Each participant was presented with 132 singing samples and instructed to listen to each complete sample before rating its 18 timbre attributes on a 1–10 scale. The audio sample order was randomized, and the descriptor order was shuffled for each sample to reduce potential anchoring effects.

The released JSON contains aggregated descriptor scores for each sample. Individual annotator ratings and per-score response counts are not included.

### Timbre descriptors

The release retains all **18 annotated descriptors**. The JSON keys are listed below exactly as stored in the file.

| JSON key | Descriptor |
| --- | --- |
| `bright` | Bright |
| `crisp` | Crisp |
| `dark` | Dark |
| `harmonize` | Harmonious |
| `hoarse` | Hoarse |
| `low` | Low |
| `magnetic` | Magnetic |
| `muddy` | Muddy |
| `pure` | Pure |
| `rich` | Rich |
| `rough` | Rough |
| `round` | Round |
| `sharp` | Sharp |
| `shriveled` | Shriveled |
| `slim` | Slim |
| `soft` | Soft |
| `thick` | Thick |
| `thin` | Thin |

**Naming note:** The descriptor called *Harmonious* in the manuscript is stored under the key `harmonize`. Use `harmonize` when reading the JSON.

The associated study selected **Bright, Thick, Soft, Pure, and Magnetic** for its prediction experiments, based on questionnaire responses about recommended descriptors and similar/opposite descriptor pairs. This five-dimensional experimental subset does not replace the full 18-descriptor annotation release.

### Inter-rater reliability

Inter-rater reliability was assessed using **ICC(3,k)**. As reported in the manuscript, all descriptors except **Dark** achieved ICC values above 0.7.

![Inter-rater reliability of the timbre descriptors](ICC_pic.png)

## Label format

`SDVED_TDA.json` is a JSON array with **132 objects**. Each object contains an `audioFile` string and 18 numeric descriptor scores.

The first record is shown below:

```json
{
  "audioFile": "audio/DH/DH_但愿人长久.wav",
  "bright": 4.1333333333,
  "crisp": 2.5384615385,
  "dark": 4.0666666667,
  "harmonize": 1.4285714286,
  "hoarse": 6.4666666667,
  "low": 4.8666666667,
  "magnetic": 2.8666666667,
  "muddy": 7.2666666667,
  "pure": 2.6666666667,
  "rich": 4.4166666667,
  "rough": 7.4166666667,
  "round": 3.1333333333,
