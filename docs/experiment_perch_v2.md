# Experiment A — BirdNET V2.4 vs Perch v2

This experiment lives only on the `experiment/perch-v2` branch. The stable
`main` pipeline is unchanged.

## Goal

Run BirdNET V2.4 and Perch v2 on the same unlabeled AudioMoth recordings and
compare their species detections without training either model.

## Setup

Install the experimental Perch runtime into the existing WildEcho environment:

```bash
uv pip install -r requirements-perch.txt
```

Perch v2 uses TensorFlow and downloads its model on first use. The model download
is roughly 380 MB.

Run the experiment:

```bash
uv run wildecho run -c configs/perch_experiment.yaml
```

## Model settings

### BirdNET baseline

- 48 kHz input
- 3-second windows
- 2-second overlap
- 1-second step
- raw candidate threshold: 0.25
- sensitivity: 1.0
- BirdNET latitude + longitude + 48-bin seasonal filter

### Perch v2

- 32 kHz input
- 5-second windows
- 4-second overlap
- 1-second step
- top 5 predictions per window
- softmax score threshold: 0.05
- CPU by default
- same BirdNET location/season candidate set, matched by exact scientific name

The two score columns are deliberately kept separate. A BirdNET confidence of
0.7 and a Perch softmax score of 0.7 are not assumed to mean the same thing.

## Shared location/season filter

BirdNET first resolves the recording date from filenames such as
`20260706_180000.WAV` into BirdNET's 48-bin seasonal week and produces the
local candidate species set from latitude, longitude, week, and the configured
frequency threshold.

Perch's 14,795 labels are then intersected with that candidate set by exact
scientific name. This removes most geographically implausible taxa and general
sound-event classes while keeping the ecological filter identical across the
two acoustic models.

The JSON records:

- `perch_location_filter_applied`
- `perch_week_48`
- `perch_candidate_species_count`
- `perch_model_species_count`
- `perch_location_candidate_coverage`

The exact-name intersection is intentionally simple for Experiment A. Taxonomic
renames between BirdNET and Perch can cause a legitimate species to be missing
from the intersection; a taxonomy crosswalk can be added in a later experiment.

## Outputs

The normal BirdNET outputs remain unchanged. When Perch is enabled WildEcho also
writes:

- `perch_detections.csv` — individual Perch detections
- `bird_model_comparison.csv` — per-recording, per-species comparison of
  BirdNET and Perch
- per-file JSON with both `bird_detections` and `perch_detections`

The comparison CSV contains:

- recording
- scientific name
- common name
- agreement: `Both`, `BirdNET only`, or `Perch only`
- BirdNET detection count
- BirdNET maximum score
- Perch detection count
- Perch maximum score
- Perch score transform

## Interpretation

Agreement means both pretrained models detected the same species in a recording.
It is useful evidence, but it is not ground-truth validation.

Experiment A is intended to answer:

1. Which species do both models detect?
2. Where do they disagree?
3. How do their score distributions differ?
4. How much extra processing time does Perch require?
5. How much of BirdNET's local species candidate set is represented in Perch?

No teacher-student training, pseudo-labeling, or model ensembling is performed in
this experiment.
