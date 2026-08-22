# WildEcho

WildEcho analyzes AudioMoth `.WAV` recordings for:

* bird detections with **BirdNET**
* human speech with **Silero VAD**
* estimated speaker count with **diarization**
* bird detections overlapping human speech
* processing time for each analysis stage
* easy-to-read CSV and JSON results

**Current version:** `v0.5.0`

## Supported systems

* Windows 10/11 — PowerShell / Windows Terminal
* Arch Linux
* macOS

---

# Installation

## Windows

Open **PowerShell/Terminal**.

Install Git:

```powershell
winget install --id Git.Git -e
```

Install FFmpeg:

```powershell
winget install --id Gyan.FFmpeg -e
```

Install `uv`:

```powershell
winget install --id astral-sh.uv -e
```

Close and reopen PowerShell/Terminal.

Check:

```powershell
git --version
uv --version
ffmpeg -version
```

---

## Arch Linux

Open **Terminal**:

```bash
sudo pacman -Syu
sudo pacman -S --needed git base-devel curl ffmpeg libsndfile
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload the shell:

```bash
source "$HOME/.local/bin/env"
```

Check:

```bash
git --version
uv --version
ffmpeg -version
```

---

## macOS

Open **Terminal**.

Install Apple Command Line Tools:

```bash
xcode-select --install
```

Install Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

On Apple Silicon Macs:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Install the required programs:

```bash
brew install git ffmpeg libsndfile
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload the shell:

```bash
source "$HOME/.local/bin/env"
```

Check:

```bash
git --version
uv --version
ffmpeg -version
```

---

# Download WildEcho

## Windows

```powershell
cd $HOME

mkdir Projects -ErrorAction SilentlyContinue
cd Projects

git clone https://github.com/hei-planet/wildecho.git
cd wildecho
```

## Linux / macOS

```bash
mkdir -p ~/Projects
cd ~/Projects

git clone https://github.com/hei-planet/wildecho.git
cd wildecho
```

---

# Install WildEcho

These commands are the same on all systems:

```bash
uv python install 3.13
uv sync --python 3.13
```

Check the installation:

```bash
uv run wildecho --version
```

Optional: run the tests:

```bash
uv run pytest
```

---

# Add your recordings

Put your AudioMoth `.WAV` files inside:

```text
data/
```

Example:

```text
wildecho/
├── data/
│   ├── 20260315_092000.WAV
│   ├── 20260315_093000.WAV
│   └── 20260315_094000.WAV
├── configs/
├── outputs/
└── ...
```

## Copy from an SD card

### Windows

If the AudioMoth SD card is `D:`:

```powershell
Copy-Item "D:\*.WAV" "data\"
```

Check how many recordings were copied:

```powershell
(Get-ChildItem "data" -Filter "*.WAV").Count
```

### Arch Linux

Find the SD card:

```bash
lsblk
```

Then copy the recordings, for example:

```bash
cp /run/media/$USER/AUDIOMOTH/*.WAV data/
```

Check the count:

```bash
find data -maxdepth 1 -type f -iname "*.wav" | wc -l
```

### macOS

List mounted drives:

```bash
ls /Volumes
```

Then copy the recordings, for example:

```bash
cp /Volumes/AUDIOMOTH/*.WAV data/
```

Check the count:

```bash
find data -maxdepth 1 -type f -iname "*.wav" | wc -l
```

> Keep the original SD-card recordings until your copied files and analysis results have been backed up.

---

# Configuration

The default configuration is:

```text
configs/default.yaml
```

The important settings are:

```yaml
input_dir: "data/"
output_dir: "outputs/"
file_glob: "*.WAV"
```

Validate it with:

```bash
uv run wildecho validate --config configs/default.yaml
```

Expected output:

```text
Config is valid
```

---

# Run WildEcho

Use the same command on Windows, Linux, and macOS:

```bash
uv run wildecho run --config configs/default.yaml
```

WildEcho will:

1. load each recording
2. detect human speech
3. run diarization only when speech is detected
4. detect birds with BirdNET
5. calculate bird/speech overlap
6. save the results

Stop the pipeline at any time with:

```text
Ctrl + C
```

---

# Live output

Example:

```text
✓ [   3/64]   4.7%  20260315_094000.WAV
  speech=no  spk=0  birds=10sp/52det
  load 0.14s  VAD 6.88s  diar 0.00s  BirdNET 10.74s
  total 17.76s  ETA 19m 46s
```

| Output             | Meaning                                              |
| ------------------ | ---------------------------------------------------- |
| `speech=no`        | No human speech detected                             |
| `speech=yes`       | Human speech detected                                |
| `spk=0`            | No speech, therefore no speakers                     |
| `spk=2`            | Two speakers estimated                               |
| `spk=?`            | Speech exists but speaker estimation was unavailable |
| `birds=10sp/52det` | 10 bird species, 52 total detections                 |
| `VAD`              | Speech-detection processing time                     |
| `diar`             | Speaker-diarization processing time                  |
| `BirdNET`          | BirdNET processing time                              |
| `total`            | Total processing time for the recording              |
| `ETA`              | Estimated remaining processing time                  |

---

# Diarization status

Successful:

```text
speech=yes  spk=2  diar=ok
```

Speech was detected and two speakers were estimated.

Too little usable speech:

```text
speech=yes  spk=?  diar=too-short
```

Speech exists, but the segments are too short for reliable speaker estimation.

Speaker embeddings unavailable:

```text
speech=yes  spk=?  diar=unavailable
```

Speech was detected, but the diarization model could not reliably estimate the speakers.

This does **not** stop the analysis. BirdNET and speech results are still saved.

---

# Results

Results are saved inside:

```text
outputs/
```

The main files are:

```text
outputs/
├── recordings.csv
├── bird_detections.csv
├── speech_segments.csv
├── summary.csv
├── pipeline.log
└── <recording-name>.json
```

## `recordings.csv`

The easiest file to use.

One row per recording with:

* filename
* duration
* speech detected
* speech duration
* estimated speakers
* diarization status
* number of bird species
* number of bird detections
* top bird
* bird/speech overlap
* processing time
* errors

## `bird_detections.csv`

One row per BirdNET detection with:

* recording
* species
* confidence
* start time
* end time
* human-speech overlap

## `speech_segments.csv`

Human-speech intervals with:

* recording
* start time
* end time
* speaker label when available

## JSON files

Each recording also gets a detailed JSON result.

---

# Open the results

## Windows

```powershell
Invoke-Item "outputs\recordings.csv"
```

Or open the entire folder:

```powershell
explorer.exe outputs
```

## Arch Linux

```bash
xdg-open outputs/recordings.csv
```

## macOS

```bash
open outputs/recordings.csv
```

---

# Separate recording batches

For different sites, weeks, or AudioMoth devices, keep each batch separate.

Example:

```text
data/
├── R4_W39/
└── R5_W39/

outputs/
├── R4_W39/
└── R5_W39/
```

Copy the default configuration:

### Windows

```powershell
Copy-Item configs\default.yaml configs\R4_W39.yaml
```

### Linux / macOS

```bash
cp configs/default.yaml configs/R4_W39.yaml
```

Edit:

```yaml
input_dir: "data/R4_W39/"
output_dir: "outputs/R4_W39/"
```

Then run:

```bash
uv run wildecho run --config configs/R4_W39.yaml
```

---

# Performance

WildEcho automatically selects reasonable performance settings.

If you want to control them manually:

## Windows PowerShell

```powershell
$env:WILDECHO_FILE_WORKERS = "2"
$env:WILDECHO_BIRDNET_THREADS = "4"

uv run wildecho run --config configs/default.yaml
```

## Linux / macOS

```bash
WILDECHO_FILE_WORKERS=2 \
WILDECHO_BIRDNET_THREADS=4 \
uv run wildecho run --config configs/default.yaml
```

Use fewer workers if the computer runs out of memory or becomes unresponsive.

---

#

# Update WildEcho

```bash
git pull --ff-only
uv sync --python 3.13
```

---

# Quick workflow

Once WildEcho is installed, the normal workflow is:

1. Copy `.WAV` recordings into `data/`
2. Check the recording count
3. Run:

```bash
uv run wildecho run --config configs/default.yaml
```

4. Open:

```text
outputs/recordings.csv
```

That's it.
