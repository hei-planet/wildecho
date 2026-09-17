# WildEcho

WildEcho analyzes AudioMoth `.WAV` recordings for:

- bird detections with **BirdNET**
- human speech with **Silero VAD**
- estimated speaker count with **diarization**
- bird detections overlapping human speech
- reproducible CSV and JSON results

**Current version:** `v0.7.0`

## Supported systems

- Windows 10/11
- Arch Linux
- macOS

## Installation

Install Git, FFmpeg, libsndfile where needed, and `uv`. Then:

```bash
git clone https://github.com/hei-planet/wildecho.git
cd wildecho
uv python install 3.13
uv sync --python 3.13
```

Check:

```bash
uv run wildecho --version
uv run pytest
```

## Recordings

Put AudioMoth WAV files in:

```text
data/
```

WildEcho recognizes timestamped filenames such as:

```text
20260315_092000.WAV
```

This is interpreted as `2026-03-15 09:20:00`. When `week_48: auto` is enabled, WildEcho converts that date to the same 48-bin seasonal week used by BirdNET. The example above resolves to **BirdNET week 10**.

## Configuration

The default configuration is:

```text
configs/default.yaml
```

The current BirdNET research settings are:

```yaml
bird_detection:
  enabled: true
  sample_rate: 48000
  min_confidence: 0.25
  sensitivity: 1.0
  overlap_sec: 2.0
  latitude: 49.4085557730695
  longitude: 8.661985343840124
  week_48: auto
  species_frequency_threshold: 0.03
  threads: auto
  batch_size: 8
```

### What these settings mean

- `min_confidence: 0.25` keeps a permissive set of raw BirdNET candidates for later species-specific validation/calibration.
- `sensitivity: 1.0` uses BirdNET's standard score transformation and should remain fixed across a dataset.
- `overlap_sec: 2.0` analyzes 3-second BirdNET windows every 1 second.
- `latitude` and `longitude` restrict the candidate species pool to the recording location.
- `week_48: auto` derives BirdNET's seasonal bin from the recording filename when possible.
- `species_frequency_threshold: 0.03` applies the BirdNET location/season species-frequency filter.

If WildEcho cannot infer the date from a filename while `week_48: auto` is selected, it logs a warning and uses BirdNET's year-round location filter rather than guessing a week.

Validate the configuration with:

```bash
uv run wildecho validate --config configs/default.yaml
```

## Run WildEcho

```bash
uv run wildecho run --config configs/default.yaml
```

WildEcho will:

1. load each recording
2. run quality control
3. resample to 16 kHz and detect human speech with Silero VAD
4. run speaker diarization only when usable speech is present
5. analyze 48 kHz audio with BirdNET
6. apply the configured location/season candidate-species filter
7. calculate bird/human-speech overlap
8. save JSON and CSV results

## BirdNET processing

With the default `overlap_sec: 2.0`, a recording is analyzed like this:

```text
0–3 s
1–4 s
2–5 s
3–6 s
...
```

This improves coverage of calls that cross the boundary of a 3-second window, at the cost of more BirdNET inference work.

BirdNET scores are detector scores, not direct probabilities that a species is present. WildEcho therefore keeps the raw threshold at `0.25`; species-specific reliability calibration should be performed from expert-validated detections rather than replacing it with one universal high cutoff.

## Location and seasonal filtering

When both coordinates are configured, WildEcho asks BirdNET for the expected species list using:

```text
latitude + longitude + BirdNET week + species-frequency threshold
```

For example:

```text
20260315_092000.WAV
        ↓
15 March 2026
        ↓
BirdNET week 10
        ↓
49.4085557730695, 8.661985343840124
        ↓
seasonally and geographically filtered candidate species
```

The resolved context is saved with the output, including:

- latitude and longitude
- BirdNET 48-week bin
- whether location filtering was applied
- candidate-species count
- species-frequency threshold
- sensitivity
- overlap

## Results

Results are written to:

```text
outputs/
├── recordings.csv
├── bird_detections.csv
├── speech_segments.csv
├── summary.csv
├── pipeline.log
└── <recording-name>.json
```

### `recordings.csv`

One row per recording with QC, speech, speaker, BirdNET context, bird summary, overlap counts, timing, and errors.

### `bird_detections.csv`

One row per retained bird detection with species, score, time interval, and human-speech overlap.

### `speech_segments.csv`

Human-speech intervals with speaker identity when diarization is available.

### Per-file JSON

The detailed JSON output includes model outputs plus the resolved BirdNET ecological settings needed to reproduce the analysis.

## Separate deployments

For another site or deployment, copy the configuration and change the coordinates rather than reusing the current site's values:

```bash
cp configs/default.yaml configs/site_02.yaml
```

Then edit:

```yaml
input_dir: "data/site_02/"
output_dir: "outputs/site_02/"

bird_detection:
  latitude: <site latitude>
  longitude: <site longitude>
```

Keep sensitivity and the other scientific settings fixed across recordings that are intended to be compared directly, unless a documented calibration protocol requires otherwise.

## Performance

WildEcho automatically chooses conservative file workers and BirdNET CPU threads. They can be overridden with:

```bash
WILDECHO_FILE_WORKERS=2 \
WILDECHO_BIRDNET_THREADS=4 \
uv run wildecho run --config configs/default.yaml
```

Use fewer workers if memory is limited.

## Update

```bash
git pull --ff-only
uv sync --python 3.13
```
