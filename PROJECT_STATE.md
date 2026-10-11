# Project State — Gaze Control

## Mission
Build a full-screen, offline, CPU-only Pygame gaze-control demo using webcam
facial landmarks. The app does not control the OS cursor or other applications.

## Constraints
- Python 3.11 and pinned packages in `requirements.txt`.
- Full-screen Pygame coordinate space; no window offsets or multi-monitor.
- Video remains local; no network calls.
- Follow `SPEC_LOCK.md`; do not implement out-of-scope behavior.

## Current status
- Stage 0: environment setup — COMPLETE
- Stage 1: landmarks — COMPLETE; user confirmed smooth tracking
- Stage 2: iris — COMPLETE; user confirmed directional response; raw gaze is
  noisy and has small dynamic range
- Stage 3: forgiving MVP calibration — IMPLEMENTED; live acceptance pending
- Stage 4: smoothing — PENDING
- Stage 5: dwell — PENDING
- Stage 6: blink — PENDING
- Stage 7: scroll — PENDING
- Stage 8: polish — PENDING

## Stage 1
- `main.py` opens a local webcam at 640x480 and processes Face Mesh with refined
  landmarks.
- User confirmed face detection and smoothly moving landmarks.

## Stage 2
- Displays refined iris centers and normalized `(gx, gy)` ratios relative to
  eye corners and eyelid opening. This rough signal is not screen coordinates.
- An open eye can provide gaze if the other is closed; gaze is unavailable when
  neither eye is usable.
- User confirmed markers and direction changes; raw frame jitter is expected
  before the later smoothing stage.

## Stage 3 architecture and implementation
- Run `.\.venv\Scripts\python.exe main.py --calibrate` for full-screen 3x3
  calibration.
- Every target shows a pulsing 3-2-1 countdown, then collects 30 valid frames.
  Valid frames require face detection, available gaze, and EAR > 0.15; head
  pose and gaze steadiness are not gates.
- A progress bar shows frame count, and every accepted frame is logged to the
  terminal. If a target cannot reach 30 frames in 20 seconds, it pauses for an
  explicit **Retry this dot** click.
- Fits the existing six-term quadratic model on all nine dots once using
  NumPy `lstsq`. Mean in-sample error below 20% of screen diagonal
  saves coefficients with `verified: true`; failure offers an explicit
  **Retry calibration** button without automatically restarting.
- Unexpected exceptions print a full traceback and leave an error screen open
  until closed. Missing, corrupt, and unverified configs cannot enable later
  interaction stages.
- Stage 3 live calibration remains pending. Procedure: `STAGE_3.md`.

## Validation
- Tests cover landmark/gaze calculations, polynomial fitting, MVP frame
  validity and acceptance, and verified-config status.
- Run `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

## Current and next work
- Current: user runs Stage 3 and confirms 30-frame collection completes, the
  fit succeeds, and `config.json` is saved.
- The read-only diagnostic can be run with
  `.venv\Scripts\python.exe main.py --diagnose` after verified calibration.
  It reports mean predictions/errors at the nine grid positions and follows
  with 20 seconds of unfiltered free-look; it never recalibrates or saves.
- Stage 4 cursor smoothing/interaction has not been implemented; there is no
  cursor-smoothing screen that calibration transitions into yet.
- Do not proceed to Stage 4+ until Stage 3 live calibration is confirmed.

## Handoff guidance
- Keep behavior offline, CPU-only, and within the locked scope.
- Update `PROGRESS.md` whenever stage status changes.
- If the environment breaks, recreate `.venv` from `requirements.txt` and
  validate imports/tests before proceeding.
