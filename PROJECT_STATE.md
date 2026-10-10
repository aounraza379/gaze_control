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
- Stage 3: calibration and cursor verification — IMPLEMENTED; live acceptance
  pending
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
- Captures only a user-specific neutral head-pose median: 0.5-second settle,
  2-second capture, at least 10 readings.
- For each dot: a pulsing 3-2-1 countdown (one second per number) with the
  instruction to look at the dot and hold steady, then a fixed 1-second
  collection window. Sensor-valid samples require detected face, gaze
  available, EAR > 0.25, and yaw/pitch each within 15 degrees of the neutral
  head-pose baseline.
- Each dot must pass gx and gy population standard deviation <= 0.02.
  Unsteady dots show a warning and repeat. Up to three retries are allowed per
  dot; a third failed retry aborts calibration.
- Fits the existing six-term quadratic model using NumPy `lstsq`.
- Reports mean predicted-position residual for each dot. Missing samples or
  residual above 15% of screen diagonal mark that dot bad; only bad dots are
  repeated, then the model and all residuals are recomputed.
- After all dot residuals pass, presents five random verification points for
  one second each. The fitted cursor is drawn live with current pixel error;
  per-point and overall mean error are reported at the end. Verification passes
  only if overall mean error is below 10% of screen diagonal.
- Only after verification passes does it save coefficients with
  `verified: true`, provided all nine dots passed steadiness. Missing, corrupt,
  and unverified configs start calibration; future interaction stages must
  require the verified marker.
- Stage 3 live calibration and cursor direction verification remain pending.
  Procedure: `STAGE_3.md`.

## Validation
- Tests cover landmark and gaze calculations, unchanged polynomial fit,
  per-dot residual selection, pixel prediction, verification pass/fail, and
  verified-config status.
- Run `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

## Current and next work
- Current: user runs Stage 3, confirms residual handling and five-point cursor
  verification, and checks top-left, center, and bottom-right direction.
- If the cursor moves opposite to gaze, use live verification evidence to
  diagnose the normalization/model sign convention; do not add another
  pre-calibration matching gate.
- Do not proceed to Stage 4+ until Stage 3 live verification passes.

## Handoff guidance
- Keep behavior offline, CPU-only, and within the locked scope.
- Update `PROGRESS.md` whenever stage status changes.
- If the environment breaks, recreate `.venv` from `requirements.txt` and
  validate imports/tests before proceeding.
