# Changelog

## 0.7.0 — 2026-09-17

### BirdNET ecology settings

- Added configurable BirdNET sensitivity and fixed the research default at `1.0`.
- Changed the default overlap to `2.0` seconds, producing 1-second steps between 3-second BirdNET windows.
- Added the current deployment coordinates: latitude `49.4085557730695`, longitude `8.661985343840124`.
- Added automatic BirdNET 48-week seasonal filtering from AudioMoth filenames such as `20260315_092000.WAV`.
- Added configurable species-frequency filtering with a default of `0.03`.
- Kept the raw BirdNET candidate threshold at `0.25` for later species-specific calibration.

### Reproducibility

- BirdNET outputs now record whether the location filter was applied, the resolved 48-week bin, candidate-species count, coordinates, sensitivity, overlap, and species-frequency threshold.
- Added validation for coordinates, sensitivity, week selection, and species-frequency threshold.
- Added regression tests confirming `20260315_092000.WAV` resolves to BirdNET week 10.

## 0.6.0 — 2026-09-17

### Reliability

- Replaced substring-based non-bird filtering with exact normalized BirdNET label matching to avoid accidentally rejecting legitimate bird names.
- Clamped final BirdNET detection timestamps to the real recording duration when the last inference window is zero-padded.
- Made BirdNET confidence validation consistent with inference: supported values are now `0.01` through `0.99` and invalid values fail instead of being silently changed.
- Added strict YAML key validation so misspelled configuration options are reported instead of silently ignored.
- Applied the same configuration validation to `wildecho run-file` as `wildecho run`.

### Developer experience

- Added GitHub Actions CI for Ruff and Pytest on Python 3.12 and 3.13.
- Added regression tests for exact bird filtering, strict configuration validation, confidence limits, and final-window timestamps.

## 0.5.0 — WildEcho

- Renamed the public project from AudioMoth Pipeline to WildEcho.
- Renamed the command-line program from `audiomoth-pipeline` to `wildecho`.
- Added `WILDECHO_*` performance environment variables while retaining compatibility with the older names.
- Prepared the repository for a clean organization-owned GitHub history.

## 0.4.1 — 2026-08-04

### Diarization reliability

- Converted empty speaker-embedding results into explicit `insufficient_speech`, `no_embeddings`, or `failed` statuses.
- Stopped reporting zero speakers when speech was detected but the speaker estimate was unavailable.
- Added a configurable minimum usable speech-segment duration and non-fatal error policy.
- Writes temporary diarization audio explicitly as PCM16 for broad embedding-runtime compatibility.
- Suppressed the ambiguous upstream warning and replaced it with a concise per-file warning in the main progress log.
- Preserved bird–human speech overlap using Silero VAD intervals whenever speaker identities are unavailable.

### Readable reports

- Added `recordings.csv`, one readable row per recording.
- Added `bird_detections.csv`, one row per bird detection.
- Added `speech_segments.csv`, one row per VAD or diarized speech segment.
- Added plain-English diarization notes and status fields to JSON and CSV output.
- Kept `summary.csv` unchanged for backward compatibility.

### Tests

- Added tests for short-speech handling, empty embedding results, VAD overlap fallback, and readable CSV generation.

## 0.4.0 — 2026-08-04

### Performance

- Switched Silero VAD to its ONNX CPU backend by default.
- Replaced the previous resampling path with direct `soxr` calls.
- Removed BirdNET temporary-WAV creation and its second audio decode.
- Added batched BirdNET inference with a configurable batch size.
- Rebuilt BirdNET's TFLite interpreter with configurable CPU threads.
- Added safe multi-process file processing with automatic CPU/RAM-aware worker selection.
- Reused the same 16 kHz buffer for VAD and diarization.
- Kept the no-speech diarization short circuit.
- Prevented nested Torch, BLAS, and numerical-library thread pools from oversubscribing the CPU.

### Reliability

- Added a safe single-worker `make run-safe` command.
- Added automatic BirdNET fallback to batch size 1 if batched tensors are unsupported.
- Added optional fingerprint-checked resume support.
- Kept summary output in deterministic input order during parallel runs.
- Corrected human-speech overlap to use interval union rather than double-counting overlapping speakers.
- Added performance configuration validation and environment overrides.

### Reporting

- Startup logs now show workers, BirdNET threads, resampler, detected CPUs, and available RAM.
- Parallel progress and ETA are based on completed files.
- Final throughput reports wall-clock performance while stage totals report cumulative worker time.

### Tests

- Added tests for batched BirdNET input, fixed final-batch padding, fast resampling, environment overrides, configuration validation, and parallel batch execution.

## 0.3.0 — 2026-08-04

- Added `make run`, colored progress, per-stage timings, ETA, `pipeline.log`, and final performance reporting.
