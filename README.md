# Gaze Control (FYP Direction 1)

Hands-free screen control via webcam gaze and blinks.
Pygame full-screen demo. Offline. CPU-only.

## Status
Stages 0–2 complete. Stage 3 uses forgiving 30-frame-per-target calibration.
Stage 4 cursor smoothing is implemented; live acceptance is pending.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Run
`.\.venv\Scripts\python.exe main.py` opens the Stage 4 full-screen smoothed
gaze cursor and requires a verified `config.json`. Run
`.venv\Scripts\python.exe main.py --calibrate` to calibrate, or
`.\.venv\Scripts\python.exe main.py --preview` to open the face/iris preview. Run
`.venv\Scripts\python.exe main.py --diagnose` to assess an already verified
calibration without changing its config.

Diagnosis presents the nine calibration-grid targets with a 3-second countdown
and 2-second prediction window per target, then prints and displays the pixel
error and signed axis biases. It ends with 20 seconds of unfiltered free-look.
It requires a verified `config.json`; it does not recalibrate or refit.

See STAGE_1.md, STAGE_2.md, STAGE_3.md, and STAGE_4.md for details and
acceptance checks. Press Q or Esc to close the cursor or preview and to cancel
calibration.

## Spec
See SPEC_LOCK.md.

## Progress
See PROGRESS.md.

## Project context
See PROJECT_STATE.md for the current handoff state, constraints, and next work.

## Testing
Run `.venv\Scripts\python.exe -m unittest discover -s tests -v` for the
hardware-independent tests. The stage-specific live acceptance check requires a
working webcam.