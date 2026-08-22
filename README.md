WildEcho

WildEcho analyzes AudioMoth .WAV recordings for:

bird detections with BirdNET
human speech with Silero VAD
estimated speaker count with diarization
bird detections overlapping human speech
processing time for each analysis stage
easy-to-read CSV and JSON results

Current version: v0.5.0

Supported systems
Windows 10/11 — PowerShell / Windows Terminal
Arch Linux
macOS
Installation
Windows

Open PowerShell/Terminal.

Install Git:

winget install --id Git.Git -e

Install FFmpeg:

winget install --id Gyan.FFmpeg -e

Install uv:

winget install --id astral-sh.uv -e

Close and reopen PowerShell/Terminal.

Check:

git --version
uv --version
ffmpeg -version
Arch Linux

Open Terminal:

sudo pacman -Syu
sudo pacman -S --needed git base-devel curl ffmpeg libsndfile

Install uv:

curl -LsSf https://astral.sh/uv/install.sh | sh

Reload the shell:

source "$HOME/.local/bin/env"

Check:

git --version
uv --version
ffmpeg -version
macOS

Open Terminal.

Install Apple Command Line Tools:

xcode-select --install

Install Homebrew:

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

On Apple Silicon Macs:

echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
eval "$(/opt/homebrew/bin/brew shellenv)"

Install the required programs:

brew install git ffmpeg libsndfile

Install uv:

curl -LsSf https://astral.sh/uv/install.sh | sh

Reload the shell:

source "$HOME/.local/bin/env"

Check:

git --version
uv --version
ffmpeg -version
Download WildEcho
Windows
cd $HOME

mkdir Projects -ErrorAction SilentlyContinue
cd Projects

git clone https://github.com/hei-planet/wildecho.git
cd wildecho
Linux / macOS
mkdir -p ~/Projects
cd ~/Projects

git clone https://github.com/hei-planet/wildecho.git
cd wildecho
Install WildEcho

These commands are the same on all systems:

uv python install 3.13
uv sync --python 3.13

Check the installation:

uv run wildecho --version

Optional: run the tests:

uv run pytest
Add your recordings

Put your AudioMoth .WAV files inside:

data/

Example:

wildecho/
├── data/
│   ├── 20260315_092000.WAV
│   ├── 20260315_093000.WAV
│   └── 20260315_094000.WAV
├── configs/
├── outputs/
└── ...
Copy from an SD card
Windows

If the AudioMoth SD card is D::

Copy-Item "D:\*.WAV" "data\"

Check how many recordings were copied:

(Get-ChildItem "data" -Filter "*.WAV").Count
Arch Linux

Find the SD card:

lsblk

Then copy the recordings, for example:

cp /run/media/$USER/AUDIOMOTH/*.WAV data/

Check the count:

find data -maxdepth 1 -type f -iname "*.wav" | wc -l
macOS

List mounted drives:

ls /Volumes

Then copy the recordings, for example:

cp /Volumes/AUDIOMOTH/*.WAV data/

Check the count:

find data -maxdepth 1 -type f -iname "*.wav" | wc -l

Keep the original SD-card recordings until your copied files and analysis results have been backed up.

Configuration

The default configuration is:

configs/default.yaml

The important settings are:

input_dir: "data/"
output_dir: "outputs/"
file_glob: "*.WAV"

Validate it with:

uv run wildecho validate --config configs/default.yaml

Expected output:

Config is valid
Run WildEcho

Use the same command on Windows, Linux, and macOS:

uv run wildecho run --config configs/default.yaml

WildEcho will:

load each recording
detect human speech
run diarization only when speech is detected
detect birds with BirdNET
calculate bird/speech overlap
save the results

Stop the pipeline at any time with:

Ctrl + C
Live output

Example:

✓ [   3/64]   4.7%  20260315_094000.WAV
  speech=no  spk=0  birds=10sp/52det
  load 0.14s  VAD 6.88s  diar 0.00s  BirdNET 10.74s
  total 17.76s  ETA 19m 46s
Output	Meaning
speech=no	No human speech detected
speech=yes	Human speech detected
spk=0	No speech, therefore no speakers
spk=2	Two speakers estimated
spk=?	Speech exists but speaker estimation was unavailable
birds=10sp/52det	10 bird species, 52 total detections
VAD	Speech-detection processing time
diar	Speaker-diarization processing time
BirdNET	BirdNET processing time
total	Total processing time for the recording
ETA	Estimated remaining processing time
Diarization status

Successful:

speech=yes  spk=2  diar=ok

Speech was detected and two speakers were estimated.

Too little usable speech:

speech=yes  spk=?  diar=too-short

Speech exists, but the segments are too short for reliable speaker estimation.

Speaker embeddings unavailable:

speech=yes  spk=?  diar=unavailable

Speech was detected, but the diarization model could not reliably estimate the speakers.

This does not stop the analysis. BirdNET and speech results are still saved.

Results

Results are saved inside:

outputs/

The main files are:

outputs/
├── recordings.csv
├── bird_detections.csv
├── speech_segments.csv
├── summary.csv
├── pipeline.log
└── <recording-name>.json
recordings.csv

The easiest file to use.

One row per recording with:

filename
duration
speech detected
speech duration
estimated speakers
diarization status
number of bird species
number of bird detections
top bird
bird/speech overlap
processing time
errors
bird_detections.csv

One row per BirdNET detection with:

recording
species
confidence
start time
end time
human-speech overlap
speech_segments.csv

Human-speech intervals with:

recording
start time
end time
speaker label when available
JSON files

Each recording also gets a detailed JSON result.

Open the results
Windows
Invoke-Item "outputs\recordings.csv"

Or open the entire folder:

explorer.exe outputs
Arch Linux
xdg-open outputs/recordings.csv
macOS
open outputs/recordings.csv
Separate recording batches

For different sites, weeks, or AudioMoth devices, keep each batch separate.

Example:

data/
├── R4_W39/
└── R5_W39/

outputs/
├── R4_W39/
└── R5_W39/

Copy the default configuration:

Windows
Copy-Item configs\default.yaml configs\R4_W39.yaml
Linux / macOS
cp configs/default.yaml configs/R4_W39.yaml

Edit:

input_dir: "data/R4_W39/"
output_dir: "outputs/R4_W39/"

Then run:

uv run wildecho run --config configs/R4_W39.yaml
Performance

WildEcho automatically selects reasonable performance settings.

If you want to control them manually:

Windows PowerShell
$env:WILDECHO_FILE_WORKERS = "2"
$env:WILDECHO_BIRDNET_THREADS = "4"

uv run wildecho run --config configs/default.yaml
Linux / macOS
WILDECHO_FILE_WORKERS=2 \
WILDECHO_BIRDNET_THREADS=4 \
uv run wildecho run --config configs/default.yaml

Use fewer workers if the computer runs out of memory or becomes unresponsive.

Update WildEcho
git pull --ff-only
uv sync --python 3.13
Quick workflow

Once WildEcho is installed, the normal workflow is:

Copy .WAV recordings into data/
Check the recording count
Run:
uv run wildecho run --config configs/default.yaml
Open:
outputs/recordings.csv

That's it.
