# Stage 3 — Calibration and Cursor Verification

## Goal
Fit the locked six-term quadratic mapping from normalized gaze `(gx, gy)` to
full-screen pixels. Validate each calibration dot, repeat only bad dots, then
verify the fitted cursor mapping at five additional screen positions before
marking the config usable by later interaction stages.

## Run

```powershell
.venv\Scripts\python.exe main.py --calibrate
```

Calibration begins with a center reference and a neutral head-pose capture:
0.5 seconds to settle, then 2 seconds of readings (at least 10) to set the
user-specific yaw/pitch baseline.

For each of the nine grid dots, look at the dot while keeping your head steady.
The app displays a large pulsing countdown: **3**, **2**, **1**, one second
each. Under the dot it says, "Look at this dot and hold your gaze steady."
Sampling begins only after the countdown finishes, then runs for a fixed
one-second window. Sensor-valid frames require a detected face, available gaze,
average EAR above 0.25, and yaw/pitch each within 15 degrees of the captured
neutral head pose. The HUD reports the active sensor checks and sample count.

After each window, the app checks population standard deviation independently
for gx and gy. Both must be at most 0.02. If either exceeds 0.02, or no valid
gaze samples were captured, the dot fails and the app displays:
"Gaze was not steady — holding this dot again." It repeats that dot with a new
countdown. Up to three retries are allowed per dot; if the third retry also
fails, calibration aborts with a clear message and no verified config is saved.

After all dots have samples, the app fits the unchanged six-term quadratic
model and reports each dot's mean pixel residual. A dot is bad if it has no
valid samples or its residual is greater than 15% of the screen diagonal. Only
bad dots are presented again with the same settle and collection timings. The
model is refitted and all dot residuals are checked again after each redo pass.
Each recollected dot must also pass the steadiness check before refitting.

When all dot residuals pass, a cursor verification screen presents five random
points inset from screen edges, one at a time for one second each. Look at each
target. A crosshair shows the fitted model's predicted cursor position live,
with a line to the target and the current pixel error. At the end, per-point
and overall mean error are reported. Overall verification passes when mean
error is below 10% of the screen diagonal. Only then is `config.json` saved
with `verified: true`; a failed or cancelled run does not mark the calibration
usable. Press **Q** or **Esc** to quit. On a failed verification, press **R** to
restart calibration.

## Accuracy definitions

The model uses NumPy `lstsq` and the existing six-column design matrix:
`1, gx, gy, gx^2, gy^2, gx*gy`. Per-dot residuals are mean Euclidean pixel
distances between each sample's predicted position and its target. The
verification error is the mean valid per-frame pixel distance from the live
predicted cursor to the current verification target. The 10% verification
threshold uses the screen diagonal, consistent with the locked calibration
accuracy metric.

The five-point verification is the behavioral check for gaze-to-screen
direction: looking upper-left, center, lower-right, and elsewhere should place
the predicted marker correspondingly. It is distinct from the in-sample
per-dot residual check.

## Acceptance check

1. Run the command above; all nine dots should be sampled without any
   pre-calibration direction waiting.
2. Confirm the residual report marks passing dots OK and identifies only bad
   dots for redo.
3. Confirm each dot shows the 3–2–1 countdown and the steady-gaze instruction;
   deliberately move your eyes during one sample to confirm an unstable dot
   repeats, and that repeated failure eventually aborts without saving.
4. Confirm the five-point screen shows a live predicted cursor crosshair,
   pixel error, and final mean-error pass/fail.
5. On pass, confirm `config.json` contains two six-number coefficient arrays
   and `verified: true`. A failed or cancelled verification must not be used by
   later interaction stages.
6. Run `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

Stage 3 is complete only after live cursor verification passes on this machine.

## Troubleshooting

If the fitted cursor moves opposite to gaze, record the predicted positions and
the gaze direction at the verification points. Diagnose the measured
normalization/model sign convention from those observations. If verification
fails, the configuration remains unverified and must not enable later
interaction stages.
