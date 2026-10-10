# Direction 1 — Spec Lock

## Scope
- Pygame demo only. NOT OS cursor. NOT other apps.
- Pygame runs FULL-SCREEN. Calibration space = app space = screen pixels.
- Offline only. No network calls. No video leaves the machine.

## Tech Stack (pinned)
- Python 3.11
- opencv-python==4.10.0.84
- mediapipe==0.10.14
- numpy==1.26.4
- pygame==2.6.1
- (python-docx is only for make_fyp_docs.py, not the app)

## Coordinate System
- Screen pixels, origin top-left.
- Calibration maps (gx, gy) → (screen_x, screen_y) directly.
- No window offset. Full-screen guarantees this.

## Calibration
- 9 dots, 3x3 grid, shown one at a time.
- Each dot: show a pulsing 3, 2, 1 countdown (1 s each), then collect 1.0 s
  of samples.
- Before the nine targets, show a center reference dot, allow 0.5 s to settle,
  then capture a 2 s neutral head-pose baseline while the user faces the camera
  naturally; use the median of at least 10 yaw/pitch readings. Collect and
  assess gaze readings at the prompted calibration dots.
- Sample = one frame where: face detected AND EAR > 0.25 AND
  |yaw - baseline_yaw| < 15° AND |pitch - baseline_pitch| < 15° AND gaze
  available.
- Each target displays "Look at this dot and hold your gaze steady" during the
  countdown. Sampling starts only after the countdown.
- Compute the population standard deviation of gx and gy over each dot's
  collected samples. If either axis standard deviation exceeds 0.02, or if no
  valid samples were collected, fail and repeat that dot with a gaze-not-steady
  warning. Allow up to 3 retries per dot; if the third retry also fails
  steadiness, abort calibration and do not save a verified config.
- Fit: screen_x = c0 + c1*gx + c2*gy + c3*gx^2 + c4*gy^2 + c5*gx*gy.
- Same form for screen_y with coefficients d0..d5.
- Fit method: numpy.linalg.lstsq on design matrix A of shape (N, 6).
- After fitting, compute mean pixel residual for each dot from all its samples.
  A dot is bad if it has no valid samples or its mean residual exceeds 15% of
  the screen diagonal. Redo only bad dots, then refit and reevaluate.
- Once all dot residuals pass, run cursor verification at 5 random inset screen
  points. Show each point for 1 s, draw predicted cursor position live, and
  report per-point and overall mean pixel error. Verification passes when the
  overall mean error is below 10% of the screen diagonal.
- Save the 12 coefficients and `verified: true` only if all nine dots passed
  steadiness and cursor verification passes. Missing, corrupt, or unverified
  calibration config must not enable later interaction stages.

## Accuracy Metric (single, used everywhere)
- mean_2d_error = mean(||pred - true||) / sqrt(W^2 + H^2)
- Pass threshold: < 0.10 (10% of screen diagonal).

## Smoothing
- Exponential moving average: x_smooth = alpha * x_new + (1 - alpha) * x_old
- alpha = 0.30 default. Tunable 0.10–0.50 in settings.

## Dwell-to-Click
- Dwell time: 1.5 s default (tunable 1.0–3.0 s).
- Charge ring fills 0→360° over dwell time.
- On complete: fire click, reset, cooldown 500 ms on same button.
- Cursor leaves button: hold timer for 0.1 s grace, then reset.

## Blink Detection
- EAR = (vertical eye distances) / (horizontal eye distance), averaged over both eyes.
- Blink = EAR < 0.15 for 3-6 frames, then EAR > 0.25.
- Wink (one eye only) does NOT count.
- Double-blink: two valid blinks, second starts within 200-500 ms of first.
- Double-blink fires click on active target = button under current smoothed gaze point.
- Cooldown after double-blink: 300 ms.

## Edge Scroll
- Top zone: y < 0.10 * screen_height, x in right 300 px column.
- Bottom zone: y > 0.90 * screen_height, x in right 300 px column.
- Dwell 0.5 s in zone before scrolling begins.
- Scroll speed: 3 px per frame.
- Leave zone: stop immediately.

## Reliability
- Face lost > 5 frames: hide cursor, pause dwell, show "Face not detected".
- No clicks fire while face lost.
- config.json missing or corrupt: auto-run calibration.

## Logging (CSV, one file per session)
Columns: event_type, frame_timestamp, action_timestamp, latency_ms, target_id, success
event_type in {dwell_click, blink_click, scroll_up, scroll_down, calibration_done, face_lost, face_found}

## Performance Targets
- FPS: >= 25 mean on mid laptop (i5, no GPU).
- Latency gaze→cursor: < 150 ms.
- Latency gaze→click: < 200 ms.

## Test Protocol (5 users)
- Each user: 1 calibration, then 10 dwell-clicks, 10 double-blinks, 10 scroll-ups, 10 scroll-downs.
- Log every attempt.
- Post-test survey: 1-5 "how magical did it feel" rating.
- Aggregate: success rate per action, mean latency, mean FPS, mean magic rating.

## Out of Scope (do NOT build)
- OS cursor control
- Multi-monitor
- Mobile / web
- Custom ML model training
- User accounts / cloud sync
- Any feature not listed above