import argparse
import sys
from collections.abc import Iterable

import cv2
import mediapipe as mp
import numpy as np


def draw_landmarks(
    frame: np.ndarray,
    points: Iterable[tuple[float, float]],
) -> None:
    height, width = frame.shape[:2]
    for x, y in points:
        pixel_x = min(max(round(x * width), 0), width - 1)
        pixel_y = min(max(round(y * height), 0), height - 1)
        cv2.circle(
            frame,
            (pixel_x, pixel_y),
            1,
            (0, 255, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


def run(camera_index: int = 0) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check camera permissions, "
            "close other camera apps, or try another --camera-index."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Failed to read a frame from the camera.")

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb_frame)
                faces = result.multi_face_landmarks

                if faces:
                    face = faces[0]
                    draw_landmarks(
                        frame,
                        ((landmark.x, landmark.y) for landmark in face.landmark),
                    )
                    status = f"Face detected: {len(face.landmark)} landmarks"
                else:
                    status = "No face detected"

                cv2.putText(
                    frame,
                    status,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if faces else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Press Q or Esc to quit",
                    (12, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Gaze Control - Face Landmarks", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview webcam face landmarks for Gaze Control Stage 1."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index to open (default: 0).",
    )
    args = parser.parse_args()

    try:
        run(args.camera_index)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
