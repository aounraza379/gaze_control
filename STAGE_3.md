# Stage 3 — Forgiving MVP Calibration

## Goal
Fit the six-term quadratic mapping from normalized gaze `(gx, gy)` to
full-screen pixels using a forgiving, fixed-sample workflow.

## Run

```powershell
.venv\Scripts\python.exe main.py --calibrate
```

To run the read-only diagnostic against a verified calibration:

```powershell
.venv\Scripts\python.exe main.py --diagnose
```

Diagnosis uses the saved coefficients without recalibration or refitting. It
shows each of the nine grid targets with a 3-second countdown followed by
2 seconds of live cursor prediction, then displays and prints the target
predictions, pixel errors, mean error, and signed X/Y bias. A 20-second
unfiltered free-look follows. The command exits with a clear error if the
config is missing, corrupt, or not verified.

For each of the nine grid targets, look at the dot through the pulsing
**3–2–1** countdown. The app then collects 30 frames with detected face,
available gaze, and average EAR above 0.15. Head pose and gaze steadiness do
not gate samples. Invalid frames are skipped. The screen shows live progress,
for example **Collecting: 15/30 frames**, and every accepted frame is logged
to the terminal.

If a target cannot collect 30 valid frames within 20 seconds, calibration
pauses and offers **Retry this dot**. It does not automatically repeat.

After all nine targets have 30 frames, the app runs the existing six-term
quadratic `numpy.linalg.lstsq` fit exactly once. It computes mean in-sample
2D pixel error normalized by screen diagonal. Error below 20% passes and saves
the coefficients with `verified: true`. Otherwise the screen says
**Calibration Failed, please try again** and offers an explicit
**Retry calibration** button that restarts all targets; no automatic loop
occurs. On success, the result remains visible until **Q** or **Esc** is
pressed.

Unexpected runtime errors print a full traceback in the terminal and leave an
error screen open until **Q**, **Esc**, or the window close event.

## Accuracy definitions

The fit uses the six-column design matrix `1, gx, gy, gx^2, gy^2, gx*gy`.
MVP accuracy is the mean Euclidean distance between fitted predictions and
their paired calibration targets, divided by the screen diagonal. The Stage 3
pass threshold is strictly below 20%.

## Acceptance check

1. Run the command above and follow each of the nine targets.
2. Confirm each target's frame count reaches 30 with a visible progress bar
   and matching terminal logs.
3. Confirm a stalled target stops after 20 seconds and waits for **Retry this
   dot**, rather than looping.
4. After target 9, confirm there is one fit and one mean-error result. On pass,
   verify `config.json` contains two six-number coefficient arrays and
   `verified: true`. On failure, use the explicit retry button if desired.
5. Run `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

## Troubleshooting

If a target times out, check that the face is visible, eyes are open, and the
preview reports normalized gaze values. Calibration can be cancelled at any
time with **Q** or **Esc**.
