# Gaze Control (FYP Direction 1)

Hands-free screen control via webcam gaze and blinks.
Pygame full-screen demo. Offline. CPU-only.

## Status
Stage 0/8 complete. Stage 1 implementation is ready for live webcam acceptance
testing.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Run
python main.py

Stage 1 run and acceptance steps are in STAGE_1.md. Press Q or Esc in the
preview to close it.

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