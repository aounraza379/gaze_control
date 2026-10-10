# Progress
- [x] Stage 0 — env setup — PASS 2026-10-10
- [x] Stage 1 — landmarks — PASS 2026-10-10 (live webcam acceptance confirmed)
- [x] Stage 2 — iris — PASS 2026-10-10 (user confirmed directional values; raw jitter expected)
- [ ] Stage 3 — calibration and cursor verification — IN PROGRESS (live acceptance pending)
- [ ] Stage 4 — smoothing — PENDING
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
- Captures a user-specific neutral head-pose median after 0.5 seconds settle
  and 2 seconds capture (minimum 10 readings).
- Every target displays a pulsing 3-2-1 countdown (one second per number) and
  the instruction to look at the dot and hold gaze steady, then collects for a
  fixed one-second window. Sensor-valid samples require detected face,
  available gaze, EAR > 0.25, and yaw/pitch within 15 degrees of neutral
  baseline.
- Each dot must have gx and gy population standard deviation at most 0.02.
  Unsteady dots show a warning and repeat with a fresh countdown; after three
  retries for the same dot, calibration aborts without saving verified config.
- Fits the existing six-term quadratic model with `numpy.linalg.lstsq`.
- Reports each dot's mean pixel residual. Missing samples or residual above 15%
  of screen diagonal mark the dot bad; only bad dots are repeated before
  refitting.
- After per-dot checks pass, presents five random verification points. The live
  predicted cursor and pixel error are shown; final mean error must be below
  10% of screen diagonal.
- Saves coefficients with `verified: true` only when all dots passed steadiness
  and verification passes.
  Missing, corrupt, or unverified config triggers calibration rather than
  enabling later interaction stages.
- Automated tests cover sensor/head-pose filters, polynomial fitting, per-dot
  residual selection, prediction, verification thresholds, config status, and
  other landmark/gaze behavior.
- Live webcam calibration and cursor verification remain pending.
- Detailed procedure: STAGE_3.md.
- The user requested and approved adding the countdown and steadiness retry
  behavior while retaining the fit, residual, and five-point verification.

## Next work
- Complete live Stage 3 and verify the cursor direction at screen corners and
  center. Diagnose sign convention from observed verification if movement is
  reversed.
- Do not start Stage 4+ interaction behavior until Stage 3 is live-verified.
