# Project State — Gaze Control

## Mission
Build a full-screen Pygame gaze-control demo for hands-free screen interaction using webcam-based facial landmark tracking. No OS cursor control, no multi-monitor support, no network calls.

## Constraints
- Python 3.11
- Full-screen app space equals screen pixels; calibration space equals app space
- Offline-only; no video leaves the machine
- Scope limited to Pygame demo only
- Out-of-scope items: OS cursor control, multi-monitor, mobile/web, custom ML, user accounts/cloud sync, unlisted features

## Source-of-truth specification
- SPEC_LOCK.md
- PROGRESS.md

## Current status
- Stage 0: environment setup — COMPLETE
- Stage 1: landmarks — IMPLEMENTATION READY; live webcam acceptance pending
- Stage 2: iris — PENDING
- Stage 3: calibration — PENDING
- Stage 4: smoothing — PENDING
- Stage 5: dwell — PENDING
- Stage 6: blink — PENDING
- Stage 7: scroll — PENDING
- Stage 8: polish — PENDING

## What Stage 0 includes
- Create a Python 3.11 virtual environment
- Install the pinned project dependencies
- Verify imports and package versions
- Keep setup instructions in the repo for future agents

## Stage 1 implementation
- `main.py` opens a local webcam, requests 640x480 frames, converts BGR to RGB,
  and processes frames with MediaPipe Face Mesh (`refine_landmarks=True`,
  `max_num_faces=1`).
- Detected landmark points are drawn on the camera image; no gaze mapping,
  calibration, cursor movement, or OS interaction is implemented.
- Run with `python main.py`; use `--camera-index N` for a different camera.
- Press Q or Esc to quit.
- Automated renderer tests: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- Stage 1 remains IN PROGRESS until the live camera acceptance checklist in
  STAGE_1.md is completed.

## Current repository snapshot
- Root: E:\bigdata\git-clone\gaze_control
- Primary docs:
  - SPEC_LOCK.md — immutable feature spec and formulas
  - PROGRESS.md — stage tracker and completion log
  - PROJECT_STATE.md — this handoff file for context continuity
- Dependency file:
  - requirements.txt

## Current and next work
- Current: run the Stage 1 live webcam acceptance check described in STAGE_1.md.
- Next after Stage 1 passes: Stage 2 iris extraction and rough gaze-point output.

## Handoff guidance for future agents
- Do not expand scope beyond the pinned spec.
- Do not implement OS cursor control or cross-app interaction.
- Keep all logic offline and CPU-only.
- Update PROGRESS.md whenever a stage changes state.
- Keep implementation within the current stage; do not proceed to Stage 2 before
  Stage 1's live acceptance check passes.
- If the environment breaks, recreate the venv from requirements.txt and re-test imports before moving on.
