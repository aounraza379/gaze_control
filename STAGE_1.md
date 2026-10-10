# Stage 1 — Face Landmark Detection

## Goal
Show a live webcam preview with MediaPipe Face Mesh landmarks drawn over the face.
This stage is a local diagnostic preview only; it does not move a cursor or control
the operating system.

## Run
From the project directory, with the project virtual environment active:

```powershell
python main.py
```

If the default camera is unavailable, try another device index:

```powershell
python main.py --camera-index 1
```

Press **Q** or **Esc** in the preview window to quit. The preview requests 640x480
frames, converts camera frames from BGR to RGB before MediaPipe processing, and
draws the refined Face Mesh landmarks in green. The status overlay shows the
detected landmark count or "No face detected".

## Acceptance check

1. Confirm the preview opens and shows the local webcam feed.
2. Confirm a face produces 478 moving landmark dots; move your head, smile, and
   blink to check that the dots track the face.
3. Confirm leaving the camera view shows "No face detected" without crashing.
4. Confirm Q and Esc both close the preview cleanly.
5. Run `.venv\Scripts\python.exe -m unittest discover -s tests -v` for
   hardware-independent renderer checks.

## Troubleshooting

- Camera unavailable: check Windows camera permission, close applications that
  may already be using it, or try another `--camera-index`.
- No face detected: improve lighting and position the face fully in view.
- Preview fails to launch: use the project's Python 3.11 virtual environment and
  install dependencies from `requirements.txt`.

Stage 1 is complete only after the live camera acceptance check is confirmed.
