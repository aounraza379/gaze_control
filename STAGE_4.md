# Stage 4 — Cursor Smoothing

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

The default launch requires a valid `config.json` with `verified: true`. If it
is missing, corrupt, or unverified, the command prints an error and exits.
Run `.\.venv\Scripts\python.exe main.py --calibrate` to create a verified
calibration before starting Stage 4.

## Behavior

The app opens full-screen, reads webcam frames, processes MediaPipe Face Mesh,
normalizes the iris gaze, and maps it to screen coordinates using the saved
polynomial coefficients. The cursor uses the exponential moving average
specified in `SPEC_LOCK.md` with alpha `0.30`; the first valid prediction
initializes the cursor position directly.

A softly outlined, glowing 12-pixel-radius dot follows the smoothed position.
An FPS counter is shown at the top-left. Once per second, the terminal prints
the current FPS and smoothed screen-coordinate gaze.

After more than five consecutive frames without a detected face, the cursor is
hidden and **Face not detected** appears at the top-center. The cursor resumes
when face and gaze tracking return. Press **Q** or **Esc** to close cleanly.
Unexpected exceptions print a full traceback and keep an error screen open
until closed.

`--preview` retains the earlier OpenCV face/landmark preview. Stage 4 does not
include dwell, blink, scroll, smoothing settings, or OS cursor control.

## Acceptance check

1. Complete verified calibration, then run the command above.
2. Confirm the full-screen cursor appears and follows gaze in the expected
   direction at top-left and bottom-right.
3. Watch for 10 seconds; confirm the cursor is visibly smoothed and not
   jittering excessively.
4. Confirm the FPS display and once-per-second console log, targeting at least
   25 FPS.
5. Move out of the camera view for more than five frames, then return; confirm
   the cursor hides, the warning appears, and tracking resumes.
6. Press **Q** or **Esc** and confirm clean shutdown.

Stage 4 remains in progress until live webcam acceptance passes.
