# Progress
- [x] Stage 0 — env setup — PASS 2026-10-10
- [x] Stage 1 — landmarks — PASS 2026-10-10 (live webcam acceptance confirmed)
- [x] Stage 2 — iris — PASS 2026-10-10 (user confirmed directional values; raw jitter expected)
- [ ] Stage 3 — forgiving MVP calibration — IN PROGRESS (live acceptance pending)
- [x] Read-only `--diagnose` mode — IMPLEMENTED; automated tests PASS (live
  webcam acceptance pending)
- [ ] Stage 4 — cursor smoothing — IMPLEMENTED; live acceptance pending
- [ ] Stage 5 — dwell — PENDING
- [ ] Stage 6 — blink — PENDING
- [ ] Stage 7 — scroll — PENDING
- [ ] Stage 8 — polish — PENDING

## Stage 0 details
- Created the Python 3.11 project environment and installed pinned packages in
  the local virtual environment.
- Stage 1 passed; Stage 2 directional iris tracking passed.

## Project handoff note
- Source-of-truth requirements: SPEC_LOCK.md.
- Current state and handoff: PROJECT_STATE.md.
- Update this file whenever a stage changes state or scope.

## Stage 1 details
- Implemented local webcam preview with MediaPipe Face Mesh and refined
  landmarks.
- User confirmed face detection and smooth-moving landmark dots.
- Run instructions: STAGE_1.md.

## Stage 2 details
- Added refined iris-center markers and normalized gaze ratios relative to eye
  corners and eyelid opening. This is a rough signal, not a screen coordinate.
- User confirmed the markers track and gaze values change directionally; raw
  frame-to-frame jitter is expected before Stage 4.
- Run instructions: STAGE_2.md.

## Stage 3 details
- `python main.py --calibrate` runs full-screen nine-point calibration.
- Each target shows a 3-2-1 countdown, then collects 30 valid face/gaze frames
  with EAR > 0.15. Head pose and steadiness are not gates. A progress bar and
  per-frame terminal logs show collection progress.
- A target times out after 20 seconds and offers an explicit retry button.
- After all targets, fits the existing six-term quadratic model once. Mean
  in-sample error below 20% of screen diagonal saves the coefficients with
  `verified: true`; failure shows an explicit full-calibration retry button.
- No automatic per-dot redo or calibration retry loop remains.
- Unexpected exceptions print a full traceback and hold an error screen open
  until the user closes it.
- Automated tests cover the relaxed frame validity and MVP threshold, along
  with polynomial fitting, config status, and landmark/gaze behavior.
- Live webcam calibration remains pending.
- Detailed procedure: STAGE_3.md.
- The user replaced the prior stricter calibration with this forgiving MVP
  workflow. See `SPEC_LOCK.md` and `STAGE_3.md` for the updated behavior.
- A read-only `--diagnose` mode measures the verified mapping at all nine grid
  targets and shows an unfiltered free-look cursor; it does not alter
  calibration or config. Automated validation passes; live webcam acceptance
  remains pending.

## Next work
- Complete live Stage 3 and confirm 30 samples are collected per dot, the fit
  runs once, and a verified config is saved below the 20% error threshold.
- Run the Stage 4 acceptance check in `STAGE_4.md`. Do not proceed to Stage 5
  until the cursor direction, smoothing, FPS, face-loss behavior, and clean
  shutdown are confirmed live.
