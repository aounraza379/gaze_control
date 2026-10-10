# Stage 2 — Iris Extraction and Rough Gaze Point

## Goal
Use the refined iris landmarks to show both iris centers and report a rough
normalized gaze value. This is not calibrated screen position and does not move
the cursor.

## Run

```powershell
python main.py
```

Look left, right, up, and down while keeping your face in view. Two magenta
circles mark the iris centers. The overlay reports `gx` and `gy`, each measured
relative to the eye corners and eyelid opening; values near `(0.5, 0.5)` mean the
iris is near the center of the visible eye. If one eye is closed, the open eye is
used. If neither eye is usable, the preview reports that gaze is unavailable.
Press **Q** or **Esc** to quit.

## Acceptance check

1. Confirm both iris markers appear on the irises and move with them.
2. Look left/right and up/down; confirm the reported gaze values change
   consistently in the corresponding direction.
3. Close one eye; confirm a gaze value remains available using the other eye.
4. Close both eyes or leave the camera view; confirm gaze is reported unavailable
   without a crash.
5. Run `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

## Limitations

- The normalized gaze value is only a rough per-eye ratio, not a screen
  coordinate; calibration is Stage 3.
- Values depend on lighting, camera placement, glasses, and face position.
- No cursor movement, calibration, or operating-system control is included.

Stage 2 is complete only after the live checks above are confirmed.
