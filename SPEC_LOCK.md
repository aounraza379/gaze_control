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
- Each dot displays a pulsing 3, 2, 1 countdown (1 s each), then collects
  30 valid frames. Show a visible count and progress bar.
- A valid frame has a detected face, available gaze, and average EAR > 0.15.
  Do not gate collection on head pose, a neutral head baseline, or gaze
  steadiness. Skip invalid frames and continue collecting.
- If 30 valid frames are not collected within 20 seconds for a dot, stop and
  show an explicit "Retry this dot" button; do not loop/retry automatically.
- Fit: screen_x = c0 + c1*gx + c2*gy + c3*gx^2 + c4*gy^2 + c5*gx*gy.
- Same form for screen_y with coefficients d0..d5.
- Fit once from all 270 samples using numpy.linalg.lstsq on design matrix A of
  shape (N, 6). Do not redo individual dots or refit after a per-dot check.
- Compute the mean in-sample 2D pixel error, normalized by screen diagonal.
  Accept the MVP calibration only when this error is below 20%.
- On fit or accuracy failure, show "Calibration Failed, please try again" and
  a button that explicitly restarts all nine dots. Do not automatically
  restart or save an unverified config.
- On pass, save the 12 coefficients and `verified: true`. Missing, corrupt, or
  unverified calibration config must not enable later interaction stages.

## Accuracy Metric (single, used everywhere)
- mean_2d_error = mean(||pred - true||) / sqrt(W^2 + H^2)
- Stage 3 MVP calibration pass threshold: < 0.20 (20% of screen diagonal).

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