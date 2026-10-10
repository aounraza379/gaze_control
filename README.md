# Gaze Control (FYP Direction 1)

Hands-free screen control via webcam gaze and blinks.
Pygame full-screen demo. Offline. CPU-only.

## Status
Stages 0–2 complete. Stage 3 now uses fixed-window calibration, per-dot fit
residual checks with bad-dot-only redo, and five-point live cursor verification.
Live acceptance is pending.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Run
`python main.py` starts calibration if `config.json` is missing, invalid, or
not verified; otherwise it opens the diagnostic preview. Use
`.venv\Scripts\python.exe main.py --calibrate` to repeat calibration, or
`python main.py --preview` to explicitly open the face/iris diagnostic.

See STAGE_1.md, STAGE_2.md, and STAGE_3.md for details and acceptance checks.
Press Q or Esc to close the preview or cancel calibration.

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