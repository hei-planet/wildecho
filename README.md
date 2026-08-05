# WildEcho

Analyze AudioMoth WAV recordings for:

- bird detections with BirdNET;
- human speech detection with Silero VAD;
- estimated speaker count with diarization;
- bird and human-speech overlap;
- per-stage processing times;
- readable CSV and JSON results.

Current version: **v0.5.0**

> **Repository maintainers:** see [`GITHUB_SETUP.md`](GITHUB_SETUP.md) to create the organization-owned repository safely.

Supported systems:

- **Arch Linux**
- **macOS**

---

## 1. Install the required programs

Choose the section for your operating system.

### Arch Linux

Open Terminal and run:

```bash
sudo pacman -Syu
sudo pacman -S --needed git base-devel curl unzip ffmpeg libsndfile
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload the shell:

```bash
source "$HOME/.local/bin/env"
```

Check the installation:

```bash
git --version
uv --version
ffmpeg -version
```

### macOS

Install Apple Command Line Tools:

```bash
xcode-select --install
```

A window will appear. Click **Install** and wait until it finishes.

Install Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Configure Homebrew.

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Install the required programs:

```bash
brew install git make ffmpeg libsndfile
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload the shell:

```bash
source "$HOME/.local/bin/env"
```

## 2. Clone the repository

The commands are the same on Arch Linux and macOS:

```bash
mkdir -p ~/Projects
cd ~/Projects

git clone https://github.com/heiplanet/wildecho.git
cd wildecho
```

## 3. Install Python and the project

Install Python 3.13:

```bash
uv python install 3.13
```

Install the project:

```bash
uv sync --python 3.13
```

Check the pipeline version:

```bash
uv run wildecho --version
```

Run the tests:

```bash
uv run pytest
```

## 4. Add AudioMoth recordings

Place the recordings inside:

```text
data/
```

Example:

```text
AudioMoth-Pipeline/
├── data/
│   ├── 20260315_092000.WAV
│   ├── 20260315_093000.WAV
│   └── 20260315_094000.WAV
├── configs/
├── outputs/
└── Makefile
```

### Copy recordings on Arch Linux

Insert the AudioMoth SD card and find its mount path:

```bash
lsblk
```

The card is usually mounted under:

```text
/run/media/<username>/<card-name>/
```

Example:

```bash
cd ~/Projects/wildecho

cp /run/media/$USER/AUDIOMOTH/*.WAV data/
```

Replace:

```text
/run/media/$USER/AUDIOMOTH
```

with the real SD-card path.

### Copy recordings on macOS

Insert the AudioMoth SD card and list mounted drives:

```bash
ls /Volumes
```

The card may appear as:

```text
AUDIOMOTH
```

Copy the recordings:

```bash
cd ~/Projects/wildecho

cp /Volumes/AUDIOMOTH/*.WAV data/
```

Replace:

```text
/Volumes/AUDIOMOTH
```

with the real SD-card path.

### Check the number of copied recordings

Run:

```bash
find data -maxdepth 1 -type f -iname "*.wav" | wc -l
```

This should match the number of recordings you expect.

> The current pipeline scans the configured folder directly. Recordings inside subfolders are not included unless `input_dir` points to the correct subfolder.

## 5. Configure the pipeline

Open the default configuration:

```bash
nano configs/default.yaml
```

The main settings are:

```yaml
input_dir: "data/"
output_dir: "outputs/"
file_glob: "*.WAV"
```

Save Nano with:

```text
Ctrl + O
Enter
Ctrl + X
```

Validate the configuration:

```bash
uv run wildecho validate --config configs/default.yaml
```

Expected output:

```text
Config is valid
```

## 6. Run the pipeline

Run:

```bash
make run
```

WildEcho will:

1. load every matching WAV file;
2. detect human speech;
3. run speaker diarization only when speech is detected;
4. run BirdNET;
5. calculate bird and human-speech overlap;
6. save CSV and JSON results.

Stop the pipeline with:

```text
Ctrl + C
```

## Live log

Example:

```text
✓ [   3/64]   4.7%  20260315_094000.WAV
speech=no  spk=0  birds=10sp/52det
load 0.14s  VAD 6.88s  diar 0.00s  BirdNET 10.74s  output 0.00s
total 17.76s  ETA 19m 46s
```

| Field | Meaning |
|---|---|
| `speech=no` | No human speech detected |
| `speech=yes` | Human speech detected |
| `spk=0` | No speech, so no speakers |
| `spk=2` | Two speakers estimated |
| `spk=?` | Speech exists, but speaker estimation was unavailable |
| `birds=10sp/52det` | 10 species and 52 bird detections |
| `VAD` | Time used for speech detection |
| `diar` | Time used for speaker diarization |
| `BirdNET` | Time used for bird analysis |
| `total` | Total processing time |
| `ETA` | Estimated remaining batch time |

## Diarization statuses

### Successful diarization

```text
speech=yes  spk=2  diar=ok
```

Speech was detected and two speakers were estimated.

### Speech too short

```text
speech=yes  spk=?  diar=too-short
```

Speech was detected, but the speech segments were too short for reliable speaker estimation.

### Embeddings unavailable

```text
speech=yes  spk=?  diar=unavailable
```

Speech was detected, but the diarization backend could not produce reliable speaker embeddings.

This does not stop BirdNET or speech detection. The recording is still included in the results.

## 7. Open the results

The output directory is:

```text
outputs/
```

### Arch Linux

Open the folder:

```bash
xdg-open outputs
```

Open the recording overview:

```bash
xdg-open outputs/recordings.csv
```

### macOS

Open the folder:

```bash
open outputs
```

Open the recording overview:

```bash
open outputs/recordings.csv
```

### Main output files

#### `recordings.csv`

One row per recording, including:

- filename;
- duration;
- speech detected;
- total speech duration;
- estimated speakers;
- diarization status;
- number of bird species;
- total bird detections;
- top bird;
- bird detections overlapping speech;
- processing time;
- errors.

#### `bird_detections.csv`

One row per BirdNET detection, including:

- recording;
- species;
- confidence;
- start time;
- end time;
- human-speech overlap.

#### `speech_segments.csv`

Speech and diarization intervals, including:

- recording;
- speech start;
- speech end;
- speaker label when available.

#### Technical files

```text
outputs/summary.csv
outputs/pipeline.log
outputs/<recording-name>.json
```

Use:

- `recordings.csv` for the simple overview;
- `bird_detections.csv` for individual bird detections;
- `speech_segments.csv` for human-speech intervals;
- JSON files for complete technical results;
- `pipeline.log` for debugging.

## 8. Process a separate recording batch

Create an input folder:

```bash
mkdir -p data/R4_W39
```

Copy recordings into it.

Arch Linux example:

```bash
cp /run/media/$USER/AUDIOMOTH/*.WAV data/R4_W39/
```

macOS example:

```bash
cp /Volumes/AUDIOMOTH/*.WAV data/R4_W39/
```

Create a separate configuration:

```bash
cp configs/default.yaml configs/R4_W39.yaml
nano configs/R4_W39.yaml
```

Change:

```yaml
input_dir: "data/R4_W39/"
output_dir: "outputs/R4_W39/"
```

Run the batch:

```bash
make run CONFIG=configs/R4_W39.yaml
```

## 9. Performance options

Run with automatic performance settings:

```bash
make run
```

Run with one safe worker:

```bash
make run-safe
```

Set the number of parallel workers:

```bash
make run WORKERS=2
```

Set workers and BirdNET threads:

```bash
make run WORKERS=2 BIRD_THREADS=4
```

Use fewer workers if the computer becomes unresponsive or runs out of memory.

## Recommended workflow

### Arch Linux

```bash
cd ~/Projects/wildecho

cp /run/media/$USER/AUDIOMOTH/*.WAV data/

find data -maxdepth 1 -type f -iname "*.wav" | wc -l

make run

xdg-open outputs/recordings.csv
```

### macOS

```bash
cd ~/Projects/wildecho

cp /Volumes/AUDIOMOTH/*.WAV data/

find data -maxdepth 1 -type f -iname "*.wav" | wc -l

make run

open outputs/recordings.csv
```

Do not delete the original SD-card recordings until the copied recordings and analysis results have been backed up.
 