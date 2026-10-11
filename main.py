import argparse
import json
import math
import random
import sys
import textwrap
import traceback
from collections.abc import Iterable
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame

IRIS_LANDMARK_INDICES = (468, 473)
EYE_LANDMARKS = (
    (33, 133, 159, 145, 468),
    (362, 263, 386, 374, 473),
)
MIN_EYE_OPENING_RATIO = 0.08
MIN_CALIBRATION_EAR = 0.15
CALIBRATION_SAMPLES_PER_DOT = 30
CALIBRATION_DOT_TIMEOUT_SECONDS = 20.0
CALIBRATION_ERROR_LIMIT = 0.20
NEUTRAL_POSE_CAPTURE_SECONDS = 2.0
MIN_NEUTRAL_POSE_SAMPLES = 10
CALIBRATION_DOT_RESIDUAL_LIMIT = 0.15
GAZE_STEADINESS_LIMIT = 0.02
DOT_COUNTDOWN_SECONDS = 3.0
MAX_DOT_RETRIES = 3
STEADINESS_WARNING_SECONDS = 2.0
VERIFICATION_POINT_COUNT = 5
VERIFICATION_DURATION_SECONDS = 1.0
EAR_EYES = (
    (33, 133, 160, 144, 158, 153),
    (362, 263, 385, 380, 387, 373),
)
FACE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)
FACE_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)
CALIBRATION_CONFIG = Path(__file__).with_name("config.json")


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


def extract_iris_centers(
    points: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if len(points) <= max(IRIS_LANDMARK_INDICES):
        return None
    return points[IRIS_LANDMARK_INDICES[0]], points[IRIS_LANDMARK_INDICES[1]]


def normalized_gaze(
    points: list[tuple[float, float]],
    frame_size: tuple[int, int],
) -> tuple[float, float] | None:
    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        return None

    iris_centers = extract_iris_centers(points)
    if iris_centers is None:
        return None

    eye_positions: list[tuple[float, float]] = []
    for corner_a, corner_b, upper, lower, iris_index in EYE_LANDMARKS:
        iris_x, iris_y = points[iris_index]
        corner_a_x, _ = points[corner_a]
        corner_b_x, _ = points[corner_b]
        _, upper_y = points[upper]
        _, lower_y = points[lower]

        eye_width = abs(corner_b_x - corner_a_x) * frame_width
        eye_opening = abs(lower_y - upper_y) * frame_height
        if eye_width == 0 or eye_opening / eye_width < MIN_EYE_OPENING_RATIO:
            continue

        eye_positions.append(
            (
                (iris_x - min(corner_a_x, corner_b_x))
                / abs(corner_b_x - corner_a_x),
                (iris_y - min(upper_y, lower_y)) / abs(lower_y - upper_y),
            )
        )

    if not eye_positions:
        return None

    return (
        sum(position[0] for position in eye_positions) / len(eye_positions),
        sum(position[1] for position in eye_positions) / len(eye_positions),
    )


def eye_aspect_ratio(
    points: list[tuple[float, float]],
    frame_size: tuple[int, int],
) -> float | None:
    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        return None

    eye_ratios: list[float] = []
    for corner_a, corner_b, upper_a, lower_a, upper_b, lower_b in EAR_EYES:
        horizontal = np.linalg.norm(
            np.array(
                (
                    (points[corner_a][0] - points[corner_b][0]) * frame_width,
                    (points[corner_a][1] - points[corner_b][1]) * frame_height,
                )
            )
        )
        if horizontal == 0:
            continue
        vertical_a = np.linalg.norm(
            np.array(
                (
                    (points[upper_a][0] - points[lower_a][0]) * frame_width,
                    (points[upper_a][1] - points[lower_a][1]) * frame_height,
                )
            )
        )
        vertical_b = np.linalg.norm(
            np.array(
                (
                    (points[upper_b][0] - points[lower_b][0]) * frame_width,
                    (points[upper_b][1] - points[lower_b][1]) * frame_height,
                )
            )
        )
        eye_ratios.append((vertical_a + vertical_b) / (2 * horizontal))

    if not eye_ratios:
        return None
    return sum(eye_ratios) / len(eye_ratios)


def calibration_frame_is_valid(
    face_detected: bool,
    gaze: tuple[float, float] | None,
    ear: float | None,
) -> bool:
    return (
        face_detected
        and gaze is not None
        and ear is not None
        and ear > MIN_CALIBRATION_EAR
    )


def calibration_mvp_passed(mean_error: float | None) -> bool:
    return (
        mean_error is not None
        and math.isfinite(mean_error)
        and mean_error < CALIBRATION_ERROR_LIMIT
    )


def relative_head_pose(
    pose: tuple[float, float],
    pose_baseline: tuple[float, float],
) -> tuple[float, float]:
    return pose[0] - pose_baseline[0], pose[1] - pose_baseline[1]


def median_head_pose(
    samples: list[tuple[float, float]],
) -> tuple[float, float] | None:
    if len(samples) < MIN_NEUTRAL_POSE_SAMPLES:
        return None
    return (
        float(np.median([sample[0] for sample in samples])),
        float(np.median([sample[1] for sample in samples])),
    )


def estimate_head_pose(
    points: list[tuple[float, float]],
    frame_size: tuple[int, int],
) -> tuple[float, float] | None:
    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        return None

    image_points = np.array(
        [
            (points[index][0] * frame_width, points[index][1] * frame_height)
            for index in FACE_LANDMARK_INDICES
        ],
        dtype=np.float64,
    )
    focal_length = float(frame_width)
    camera_matrix = np.array(
        [
            [focal_length, 0, frame_width / 2],
            [0, focal_length, frame_height / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    success, rotation_vector, _ = cv2.solvePnP(
        FACE_MODEL_POINTS,
        image_points,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    pitch, yaw = float(angles[0]), float(angles[1])
    return yaw, pitch


def calibration_design(gx: float, gy: float) -> tuple[float, ...]:
    return (1.0, gx, gy, gx * gx, gy * gy, gx * gy)


def fit_calibration(
    gaze_samples: list[tuple[float, float]],
    screen_samples: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    if len(gaze_samples) != len(screen_samples) or len(gaze_samples) < 6:
        raise ValueError("Calibration needs at least 6 paired gaze samples.")

    design = np.array(
        [calibration_design(gx, gy) for gx, gy in gaze_samples],
        dtype=np.float64,
    )
    targets = np.array(screen_samples, dtype=np.float64)
    coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank < 6:
        raise ValueError(
            "Calibration samples do not span the screen; repeat calibration "
            "while looking at every target."
        )

    return coefficients[:, 0].tolist(), coefficients[:, 1].tolist()


def mean_2d_error(
    gaze_samples: list[tuple[float, float]],
    screen_samples: list[tuple[float, float]],
    x_coefficients: list[float],
    y_coefficients: list[float],
    screen_size: tuple[int, int],
) -> float:
    if not gaze_samples or len(gaze_samples) != len(screen_samples):
        raise ValueError("Accuracy needs paired, non-empty calibration samples.")
    width, height = screen_size
    if width <= 0 or height <= 0:
        raise ValueError("Screen dimensions must be positive.")

    design = np.array(
        [calibration_design(gx, gy) for gx, gy in gaze_samples],
        dtype=np.float64,
    )
    coefficients = np.column_stack((x_coefficients, y_coefficients))
    predicted = design @ coefficients
    actual = np.array(screen_samples, dtype=np.float64)
    return float(np.linalg.norm(predicted - actual, axis=1).mean()) / math.hypot(
        width, height
    )


def fit_mvp_calibration(
    dot_gaze_samples: list[list[tuple[float, float]]],
    target_positions: list[tuple[int, int]],
    screen_size: tuple[int, int],
) -> tuple[list[float], list[float], float]:
    if len(dot_gaze_samples) != len(target_positions):
        raise ValueError("Each calibration target needs one sample bucket.")
    if any(not samples for samples in dot_gaze_samples):
        raise ValueError("Every calibration target needs valid gaze samples.")

    gaze_samples = [
        gaze
        for samples_for_dot in dot_gaze_samples
        for gaze in samples_for_dot
    ]
    screen_samples = [
        (float(target[0]), float(target[1]))
        for samples_for_dot, target in zip(dot_gaze_samples, target_positions)
        for _ in samples_for_dot
    ]
    x_coefficients, y_coefficients = fit_calibration(
        gaze_samples, screen_samples
    )
    error = mean_2d_error(
        gaze_samples,
        screen_samples,
        x_coefficients,
        y_coefficients,
        screen_size,
    )
    return x_coefficients, y_coefficients, error


def predict_screen_position(
    gaze: tuple[float, float],
    x_coefficients: list[float],
    y_coefficients: list[float],
) -> tuple[float, float]:
    design = np.array(calibration_design(*gaze), dtype=np.float64)
    coefficients = np.column_stack((x_coefficients, y_coefficients))
    predicted = design @ coefficients
    return float(predicted[0]), float(predicted[1])


def per_dot_residuals(
    dot_gaze_samples: list[list[tuple[float, float]]],
    target_positions: list[tuple[int, int]],
    x_coefficients: list[float],
    y_coefficients: list[float],
) -> list[float | None]:
    if len(dot_gaze_samples) != len(target_positions):
        raise ValueError("Each calibration dot needs one sample bucket.")

    residuals: list[float | None] = []
    for gaze_samples, target in zip(dot_gaze_samples, target_positions):
        if not gaze_samples:
            residuals.append(None)
            continue
        errors = [
            math.dist(
                predict_screen_position(gaze, x_coefficients, y_coefficients),
                target,
            )
            for gaze in gaze_samples
        ]
        residuals.append(sum(errors) / len(errors))
    return residuals


def bad_calibration_dots(
    residuals: list[float | None],
    screen_size: tuple[int, int],
) -> list[int]:
    width, height = screen_size
    if width <= 0 or height <= 0:
        raise ValueError("Screen dimensions must be positive.")
    limit = CALIBRATION_DOT_RESIDUAL_LIMIT * math.hypot(width, height)
    return [
        index
        for index, residual in enumerate(residuals)
        if residual is None or residual > limit
    ]


def gaze_standard_deviation(
    samples: list[tuple[float, float]],
) -> tuple[float, float] | None:
    if not samples:
        return None
    deviations = np.std(np.asarray(samples, dtype=np.float64), axis=0)
    return float(deviations[0]), float(deviations[1])


def gaze_is_steady(samples: list[tuple[float, float]]) -> bool:
    deviations = gaze_standard_deviation(samples)
    return (
        deviations is not None
        and deviations[0] <= GAZE_STEADINESS_LIMIT
        and deviations[1] <= GAZE_STEADINESS_LIMIT
    )


def next_dot_retry_count(retry_count: int) -> int | None:
    next_count = retry_count + 1
    return next_count if next_count <= MAX_DOT_RETRIES else None


def countdown_number(elapsed: float) -> int:
    second_index = min(2, int(max(0.0, elapsed)))
    return 3 - second_index


def verification_mean_error(errors: Iterable[float]) -> float | None:
    collected_errors = list(errors)
    if not collected_errors:
        return None
    return sum(collected_errors) / len(collected_errors)


def verification_passed(
    mean_error: float | None,
    screen_size: tuple[int, int],
) -> bool:
    width, height = screen_size
    if width <= 0 or height <= 0:
        raise ValueError("Screen dimensions must be positive.")
    return mean_error is not None and mean_error < 0.10 * math.hypot(width, height)


def calibration_acceptance_passed(
    dot_steadiness_passed: list[bool],
    verification_mean: float | None,
    screen_size: tuple[int, int],
) -> bool:
    return all(dot_steadiness_passed) and verification_passed(
        verification_mean, screen_size
    )


def verification_points(
    screen_size: tuple[int, int],
    count: int = VERIFICATION_POINT_COUNT,
) -> list[tuple[int, int]]:
    width, height = screen_size
    if width <= 0 or height <= 0:
        raise ValueError("Screen dimensions must be positive.")
    return [
        (round(random.uniform(width * 0.1, width * 0.9)),
         round(random.uniform(height * 0.1, height * 0.9)))
        for _ in range(count)
    ]


def save_calibration(
    x_coefficients: list[float],
    y_coefficients: list[float],
    *,
    verified: bool,
) -> None:
    config = {
        "screen_x_coefficients": x_coefficients,
        "screen_y_coefficients": y_coefficients,
        "verified": verified,
    }
    temporary_path = CALIBRATION_CONFIG.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(CALIBRATION_CONFIG)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def calibration_config_status() -> str:
    try:
        config = json.loads(CALIBRATION_CONFIG.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "missing"
    except json.JSONDecodeError:
        return "corrupt"

    if not isinstance(config, dict):
        return "corrupt"
    if config.get("verified") is not True:
        return "corrupt"
    for key in ("screen_x_coefficients", "screen_y_coefficients"):
        coefficients = config.get(key)
        if (
            not isinstance(coefficients, list)
            or len(coefficients) != 6
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in coefficients
            )
        ):
            return "corrupt"
    return "valid"


def draw_calibration_hud(
    screen: pygame.Surface,
    target: tuple[int, int],
    screen_size: tuple[int, int],
    face_detected: bool,
    gaze: tuple[float, float] | None,
    ear: float | None,
    pose: tuple[float, float] | None,
    pose_baseline: tuple[float, float] | None,
    valid_sample: bool,
    phase: str,
    phase_elapsed: float,
    point_samples: int,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
) -> None:
    width, height = screen_size
    card_width = min(520, width - 32)
    card_height = 205
    card_x = 16 if target[0] >= width / 2 else width - card_width - 16
    card_y = 16 if target[1] >= height / 2 else height - card_height - 16

    panel = pygame.Surface((card_width, card_height), pygame.SRCALPHA)
    panel.fill((10, 15, 25, 235))
    screen.blit(panel, (card_x, card_y))
    pygame.draw.rect(
        screen,
        (85, 100, 125),
        (card_x, card_y, card_width, card_height),
        width=1,
        border_radius=8,
    )

    white = (235, 239, 250)
    green = (115, 235, 155)
    red = (255, 125, 115)
    muted = (185, 195, 212)
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (
            "Face: FOUND" if face_detected else "Face: NOT FOUND - move into camera view",
            green if face_detected else red,
        ),
        (
            "Eye tracking: READY"
            if gaze is not None
            else "Eye tracking: NOT READY - improve lighting/eye visibility",
            green if gaze is not None else red,
        ),
    ]
    eyes_open = ear is not None and ear > 0.25
    eye_text = (
        f"Eyes: OPEN (EAR {ear:.2f}; need > 0.25)"
        if eyes_open
        else (
            f"Eyes: OPEN EYES COMFORTABLY (EAR {ear:.2f}; need > 0.25)"
            if ear is not None
            else "Eyes: NOT READ - improve lighting"
        )
    )
    lines.append((eye_text, green if eyes_open else red))

    if phase.startswith("baseline"):
        lines.append(("Capturing your natural head position; face the screen.", muted))
    elif pose is None:
        lines.append(("Head: NOT READ - face camera with chin visible", red))
    elif pose_baseline is None:
        lines.append(("Neutral head position: NOT CAPTURED", red))
    else:
        yaw, pitch = relative_head_pose(pose, pose_baseline)
        yaw_ok = abs(yaw) < 15
        pitch_ok = abs(pitch) < 15
        lines.append(
            (
                f"Head relative to neutral: yaw {yaw:+.1f}, pitch {pitch:+.1f} deg",
                green if yaw_ok and pitch_ok else red,
            )
        )

    if phase == "baseline_settle":
        phase_text = f"Get ready; head capture starts in {max(0.0, 0.5 - phase_elapsed):.1f}s"
        phase_color = muted
    elif phase == "baseline":
        phase_text = f"Capturing neutral head pose ({max(0.0, NEUTRAL_POSE_CAPTURE_SECONDS - phase_elapsed):.1f}s)"
        phase_color = muted
    elif phase == "countdown":
        phase_text = "Hold your gaze on the dot through the countdown."
        phase_color = muted
    elif phase == "collect":
        phase_text = (
            f"Sampling valid gaze for {max(0.0, 1.0 - phase_elapsed):.1f}s"
            if valid_sample
            else "Waiting for valid face, open eyes, gaze, and steady head"
        )
        phase_color = green if valid_sample else red
    elif phase == "steadiness_retry_notice":
        phase_text = "Gaze was not steady; holding this dot again."
        phase_color = red
    else:
        phase_text = phase.replace("_", " ").title()
        phase_color = muted
    lines.append((phase_text, phase_color))
    lines.append((f"Valid samples this dot: {point_samples}", white))

    y = card_y + 10
    for index, (text, color) in enumerate(lines):
        font = title_font if index == len(lines) - 2 else body_font
        screen.blit(font.render(text, True, color), (card_x + 10, y))
        y += 23 if font is title_font else 21


def run_calibration(camera_index: int = 0) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check camera permissions, "
            "close other camera apps, or try another --camera-index."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Gaze Control Calibration")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 34)
    small_font = pygame.font.Font(None, 25)
    countdown_font = pygame.font.Font(None, 144)
    hud_title_font = pygame.font.Font(None, 21)
    hud_body_font = pygame.font.Font(None, 18)
    width, height = screen.get_size()
    target_positions = [
        (round(width * x), round(height * y))
        for y in (0.1, 0.5, 0.9)
        for x in (0.1, 0.5, 0.9)
    ]
    dot_gaze_samples: list[list[tuple[float, float]]] = [
        [] for _ in target_positions
    ]
    target_index = 0
    phase_started = pygame.time.get_ticks() / 1000
    phase = "countdown"
    point_samples = 0
    pose_baseline: tuple[float, float] | None = None
    x_coefficients: list[float] | None = None
    y_coefficients: list[float] | None = None
    fit_error: float | None = None
    predicted_cursor: tuple[float, float] | None = None
    current_verification_error: float | None = None
    verification_targets: list[tuple[int, int]] = []
    verification_errors: list[list[float]] = []
    verification_index = 0
    verification_mean: float | None = None
    calibration_residuals: list[float | None] = []
    pending_bad_dots: list[int] = []
    dot_steadiness_passed = [True for _ in target_positions]
    pose: tuple[float, float] | None = None
    fit_failure_reason: str | None = None
    calibration_cycle = 1
    retry_button = pygame.Rect(width // 2 - 130, height // 2 + 55, 260, 54)
    finished = False
    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while not finished:
                now = pygame.time.get_ticks() / 1000
                elapsed = now - phase_started
                valid_sample = False
                face_detected = False
                gaze: tuple[float, float] | None = None
                ear: float | None = None
                if phase not in (
                    "calibration_result",
                    "calibration_failed",
                    "collection_timeout",
                ):
                    frame_ok, frame = capture.read()
                    if not frame_ok:
                        raise RuntimeError("Failed to read a frame from the camera.")

                    result = face_mesh.process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    faces = result.multi_face_landmarks
                    if faces:
                        face_detected = True
                        points = [
                            (landmark.x, landmark.y)
                            for landmark in faces[0].landmark
                        ]
                        gaze = normalized_gaze(
                            points, (frame.shape[1], frame.shape[0])
                        )
                        ear = eye_aspect_ratio(
                            points, (frame.shape[1], frame.shape[0])
                        )
                        valid_sample = calibration_frame_is_valid(
                            face_detected, gaze, ear
                        )
                        if phase == "collect" and valid_sample and gaze is not None:
                            if len(dot_gaze_samples[target_index]) < CALIBRATION_SAMPLES_PER_DOT:
                                dot_gaze_samples[target_index].append(gaze)
                                point_samples = len(dot_gaze_samples[target_index])
                                print(
                                    f"Collected frame {point_samples} for dot "
                                    f"{target_index + 1}/{len(target_positions)}",
                                    flush=True,
                                )

                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN
                        and event.key in (pygame.K_ESCAPE, pygame.K_q)
                    ):
                        finished = True
                    elif (
                        event.type == pygame.MOUSEBUTTONUP
                        and event.button == 1
                        and phase in ("collection_timeout", "calibration_failed")
                        and retry_button.collidepoint(event.pos)
                    ):
                        if phase == "collection_timeout":
                            dot_gaze_samples[target_index] = []
                            point_samples = 0
                        else:
                            calibration_cycle += 1
                            dot_gaze_samples = [[] for _ in target_positions]
                            target_index = 0
                        fit_failure_reason = None
                        fit_error = None
                        phase = "countdown"
                        phase_started = now
                        elapsed = 0.0

                if phase == "countdown" and elapsed >= DOT_COUNTDOWN_SECONDS:
                    phase = "collect"
                    dot_gaze_samples[target_index] = []
                    point_samples = 0
                    phase_started = now
                    elapsed = 0.0
                elif phase == "collect":
                    point_samples = len(dot_gaze_samples[target_index])
                    if point_samples >= CALIBRATION_SAMPLES_PER_DOT:
                        print(
                            f"Completed dot {target_index + 1}/"
                            f"{len(target_positions)}",
                            flush=True,
                        )
                        point_samples = 0
                        if target_index + 1 < len(target_positions):
                            target_index += 1
                            phase = "countdown"
                            phase_started = now
                        else:
                            try:
                                print(
                                    "Fitting calibration once using all nine dots.",
                                    flush=True,
                                )
                                (
                                    x_coefficients,
                                    y_coefficients,
                                    fit_error,
                                ) = fit_mvp_calibration(
                                    dot_gaze_samples,
                                    target_positions,
                                    (width, height),
                                )
                            except ValueError as error:
                                fit_failure_reason = str(error)
                                phase = "calibration_failed"
                            else:
                                print(
                                    f"Calibration mean error: {fit_error:.1%} "
                                    "(limit: <20% of screen diagonal)",
                                    flush=True,
                                )
                                if calibration_mvp_passed(fit_error):
                                    if (
                                        x_coefficients is None
                                        or y_coefficients is None
                                    ):
                                        raise RuntimeError(
                                            "Calibration coefficients are unavailable."
                                        )
                                    save_calibration(
                                        x_coefficients,
                                        y_coefficients,
                                        verified=True,
                                    )
                                    print(
                                        "Calibration passed and config.json was saved.",
                                        flush=True,
                                    )
                                    phase = "calibration_result"
                                else:
                                    fit_failure_reason = (
                                        "Average fit error was not below 20% "
                                        "of the screen diagonal."
                                    )
                                    phase = "calibration_failed"
                            phase_started = now
                    elif elapsed >= CALIBRATION_DOT_TIMEOUT_SECONDS:
                        fit_failure_reason = (
                            f"Could not collect {CALIBRATION_SAMPLES_PER_DOT} "
                            f"valid frames for dot {target_index + 1} within "
                            f"{CALIBRATION_DOT_TIMEOUT_SECONDS:.0f} seconds."
                        )
                        phase = "collection_timeout"
                        phase_started = now

                screen.fill((15, 18, 28))
                title = font.render("Gaze Calibration", True, (235, 239, 250))
                screen.blit(title, title.get_rect(center=(width // 2, 42)))
                if phase in (
                    "calibration_result",
                    "calibration_failed",
                    "collection_timeout",
                ):
                    if phase == "calibration_result":
                        result_title = "Calibration complete and saved"
                        result_detail = (
                            f"Mean fit error: {fit_error:.1%} of screen diagonal"
                            if fit_error is not None
                            else "Mean fit error unavailable"
                        )
                        result_color = (120, 235, 160)
                    else:
                        result_title = (
                            "Calibration Failed, please try again"
                            if phase == "calibration_failed"
                            else "Could not collect enough valid gaze frames"
                        )
                        result_detail = fit_failure_reason or ""
                        result_color = (255, 145, 135)
                    screen.blit(
                        font.render(result_title, True, result_color),
                        font.render(result_title, True, result_color).get_rect(
                            center=(width // 2, height // 2 - 45)
                        ),
                    )
                    detail = small_font.render(
                        result_detail, True, (220, 225, 235)
                    )
                    screen.blit(
                        detail,
                        detail.get_rect(center=(width // 2, height // 2 + 5)),
                    )
                    if phase != "calibration_result":
                        pygame.draw.rect(
                            screen, (50, 125, 190), retry_button, border_radius=8
                        )
                        button_text = (
                            "Retry this dot"
                            if phase == "collection_timeout"
                            else "Retry calibration"
                        )
                        rendered_button = small_font.render(
                            button_text, True, (255, 255, 255)
                        )
                        screen.blit(
                            rendered_button,
                            rendered_button.get_rect(center=retry_button.center),
                        )
                    else:
                        screen.blit(
                            small_font.render(
                                "Press Q or Esc to close.", True, (195, 204, 220)
                            ),
                            small_font.render(
                                "Press Q or Esc to close.", True, (195, 204, 220)
                            ).get_rect(center=(width // 2, height // 2 + 70)),
                        )
                elif phase in ("countdown", "collect"):
                    pygame.draw.circle(
                        screen, (70, 190, 255), target_positions[target_index], 14
                    )
                    pygame.draw.circle(screen, (235, 245, 255),
                                       target_positions[target_index], 4)
                    progress = f"Target {target_index + 1}/9"
                    screen.blit(
                        small_font.render(progress, True, (235, 239, 250)),
                        (24, 28),
                    )
                    if phase == "countdown":
                        number_surface = countdown_font.render(
                            str(countdown_number(elapsed)),
                            True,
                            (255, 245, 210),
                        )
                        pulse = 1.0 + 0.12 * abs(math.sin(now * math.pi))
                        countdown_surface = pygame.transform.smoothscale(
                            number_surface,
                            (
                                round(number_surface.get_width() * pulse),
                                round(number_surface.get_height() * pulse),
                            ),
                        )
                        screen.blit(
                            countdown_surface,
                            countdown_surface.get_rect(
                                center=target_positions[target_index]
                            ),
                        )
                        notice = "Look at the target. Collection starts after countdown."
                    else:
                        notice = (
                            f"Collecting: {point_samples}/"
                            f"{CALIBRATION_SAMPLES_PER_DOT} valid frames "
                            f"({max(0, int(CALIBRATION_DOT_TIMEOUT_SECONDS - elapsed))}s)"
                        )
                    notice_surface = small_font.render(
                        notice,
                        True,
                        (195, 204, 220),
                    )
                    notice_y = min(
                        height - notice_surface.get_height() - 12,
                        target_positions[target_index][1] + 34,
                    )
                    notice_x = min(
                        max(
                            target_positions[target_index][0],
                            notice_surface.get_width() // 2 + 12,
                        ),
                        width - notice_surface.get_width() // 2 - 12,
                    )
                    screen.blit(
                        notice_surface,
                        notice_surface.get_rect(
                            center=(notice_x, notice_y)
                        ),
                    )
                    bar_width = min(420, width - 60)
                    bar_rect = pygame.Rect(
                        (width - bar_width) // 2, height - 75, bar_width, 20
                    )
                    pygame.draw.rect(
                        screen, (65, 75, 95), bar_rect, border_radius=8
                    )
                    pygame.draw.rect(
                        screen,
                        (90, 205, 150),
                        pygame.Rect(
                            bar_rect.x,
                            bar_rect.y,
                            round(
                                bar_width
                                * point_samples
                                / CALIBRATION_SAMPLES_PER_DOT
                            ),
                            bar_rect.height,
                        ),
                        border_radius=8,
                    )
                    sensor_text = (
                        "Face and eyes ready"
                        if valid_sample
                        else (
                            "Looking for face, visible gaze, and open eyes"
                            if phase == "collect"
                            else " "
                        )
                    )
                    screen.blit(
                        small_font.render(
                            sensor_text, True, (195, 204, 220)
                        ),
                        small_font.render(
                            sensor_text, True, (195, 204, 220)
                        ).get_rect(center=(width // 2, height - 105)),
                    )
                elif phase == "residual_report":
                    report_title = (
                        "Per-target fit residuals"
                        if pending_bad_dots
                        else "All target residuals are within the limit"
                    )
                    title_surface = font.render(
                        report_title, True, (235, 239, 250)
                    )
                    screen.blit(
                        title_surface,
                        title_surface.get_rect(
                            center=(width // 2, height // 2 - 190)
                        ),
                    )
                    diagonal = math.hypot(width, height)
                    residual_limit = CALIBRATION_DOT_RESIDUAL_LIMIT * diagonal
                    for index, residual in enumerate(calibration_residuals):
                        is_bad = index in pending_bad_dots
                        residual_text = (
                            f"Target {index + 1}: no valid samples - BAD"
                            if residual is None
                            else (
                                f"Target {index + 1}: {residual:.0f} px "
                                f"({residual / diagonal:.1%})"
                                + (" - REDO" if is_bad else " - OK")
                            )
                        )
                        rendered = small_font.render(
                            residual_text,
                            True,
                            (255, 130, 120) if is_bad else (120, 235, 160),
                        )
                        screen.blit(
                            rendered,
                            rendered.get_rect(
                                center=(width // 2, height // 2 - 130 + index * 30)
                            ),
                        )
                    if fit_error is not None:
                        summary = (
                            f"Overall fit error: {fit_error:.1%}; "
                            f"per-dot redo limit: {residual_limit:.0f} px (15%)"
                        )
                        rendered = small_font.render(
                            summary, True, (195, 204, 220)
                        )
                        screen.blit(
                            rendered,
                            rendered.get_rect(
                                center=(width // 2, height // 2 + 155)
                            ),
                        )
                    next_step = (
                        "Only the marked targets will be repeated."
                        if pending_bad_dots
                        else "Starting five-point live cursor verification."
                    )
                    rendered = small_font.render(
                        next_step, True, (195, 204, 220)
                    )
                    screen.blit(
                        rendered,
                        rendered.get_rect(
                            center=(width // 2, height // 2 + 190)
                        ),
                    )
                elif phase == "verification":
                    verification_target = verification_targets[verification_index]
                    pygame.draw.circle(screen, (255, 190, 70),
                                       verification_target, 14, width=2)
                    pygame.draw.circle(screen, (255, 235, 190),
                                       verification_target, 4)
                    if predicted_cursor is not None:
                        cursor_x = min(max(round(predicted_cursor[0]), 0), width - 1)
                        cursor_y = min(max(round(predicted_cursor[1]), 0), height - 1)
                        cursor_position = (cursor_x, cursor_y)
                        pygame.draw.line(screen, (105, 255, 165),
                                         verification_target, cursor_position, 1)
                        pygame.draw.circle(screen, (105, 255, 165),
                                           cursor_position, 10, width=2)
                        pygame.draw.line(screen, (105, 255, 165),
                                         (cursor_x - 7, cursor_y),
                                         (cursor_x + 7, cursor_y), 2)
                        pygame.draw.line(screen, (105, 255, 165),
                                         (cursor_x, cursor_y - 7),
                                         (cursor_x, cursor_y + 7), 2)
                        error_text = (
                            f"Live error: {current_verification_error:.0f} px"
                        )
                    else:
                        error_text = "Live cursor unavailable - fix red sensor checks"
                    screen.blit(
                        font.render(
                            f"Cursor verification {verification_index + 1}/"
                            f"{len(verification_targets)}",
                            True,
                            (235, 239, 250),
                        ),
                        (24, 24),
                    )
                    screen.blit(
                        small_font.render(
                            f"Look at the target for 1 second. {error_text}",
                            True,
                            (195, 204, 220),
                        ),
                        (24, 62),
                    )
                    draw_calibration_hud(
                        screen,
                        verification_target,
                        (width, height),
                        face_detected,
                        gaze,
                        ear,
                        pose,
                        pose_baseline,
                        valid_sample,
                        "verification",
                        elapsed,
                        len(verification_errors[verification_index]),
                        hud_title_font,
                        hud_body_font,
                    )
                elif phase in ("fit_failed", "retry_limit_failed"):
                    failure_title = (
                        "Calibration aborted: gaze was not steady"
                        if phase == "retry_limit_failed"
                        else "Calibration fit is not solvable"
                    )
                    failure_title_surface = font.render(
                        failure_title, True, (255, 130, 120)
                    )
                    fit_failure_text = (
                        (
                            f"Calibration aborted: target {target_index + 1} "
                            f"remained unsteady after {MAX_DOT_RETRIES} retries. "
                            "No verified config was saved. Press Q or Esc to quit."
                        )
                        if phase == "retry_limit_failed"
                        else (
                            f"{fit_failure_reason or 'Calibration fit failed.'} "
                            "Press R to restart, or Q to quit."
                        )
                    )
                    screen.blit(
                        failure_title_surface,
                        failure_title_surface.get_rect(
                            center=(width // 2, height // 2 - 35)
                        ),
                    )
                    screen.blit(
                        small_font.render(
                            fit_failure_text,
                            True,
                            (235, 190, 185),
                        ),
                        small_font.render(
                            fit_failure_text,
                            True,
                            (235, 190, 185),
                        ).get_rect(center=(width // 2, height // 2 + 15)),
                    )
                elif phase == "verification_result":
                    passed = calibration_acceptance_passed(
                        dot_steadiness_passed,
                        verification_mean,
                        (width, height),
                    )
                    result_color = (120, 235, 160) if passed else (255, 130, 120)
                    result_title = (
                        "Calibration verified and saved"
                        if passed
                        else "Cursor verification failed - config not saved"
                    )
                    screen.blit(
                        font.render(result_title, True, result_color),
                        font.render(result_title, True, result_color).get_rect(
                            center=(width // 2, height // 2 - 130)
                        ),
                    )
                    for index, errors in enumerate(verification_errors):
                        point_mean = verification_mean_error(errors)
                        point_text = (
                            f"Point {index + 1}: {point_mean:.0f} px mean error"
                            if point_mean is not None
                            else f"Point {index + 1}: no valid gaze samples"
                        )
                        rendered = small_font.render(
                            point_text, True, (220, 225, 235)
                        )
                        screen.blit(
                            rendered,
                            rendered.get_rect(
                                center=(width // 2, height // 2 - 75 + index * 30)
                            ),
                        )
                    diagonal = math.hypot(width, height)
                    overall_text = (
                        f"Mean error: {verification_mean:.0f} px "
                        f"({verification_mean / diagonal:.1%} of screen diagonal); "
                        "pass limit < 10%"
                        if verification_mean is not None
                        else "Mean error unavailable: no valid gaze samples"
                    )
                    rendered = small_font.render(
                        overall_text, True, result_color
                    )
                    screen.blit(
                        rendered,
                        rendered.get_rect(center=(width // 2, height // 2 + 60)),
                    )
                    action = (
                        "Press Q or Esc to close."
                        if passed
                        else "Press R to redo calibration, or Q/Esc to quit."
                    )
                    rendered = small_font.render(
                        action, True, (195, 204, 220)
                    )
                    screen.blit(
                        rendered,
                        rendered.get_rect(center=(width // 2, height // 2 + 105)),
                    )
                else:
                    screen.blit(
                        font.render(
                            f"Calibration pass {calibration_cycle} complete",
                            True,
                            (235, 239, 250),
                        ),
                        font.render(
                            f"Calibration pass {calibration_cycle} complete",
                            True,
                            (235, 239, 250),
                        ).get_rect(center=(width // 2, height // 2)),
                    )
                pygame.display.flip()
                clock.tick(30)
    finally:
        capture.release()
        pygame.quit()


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
                    points = [(landmark.x, landmark.y) for landmark in face.landmark]
                    draw_landmarks(
                        frame,
                        points,
                    )
                    status = f"Face detected: {len(face.landmark)} landmarks"
                    iris_centers = extract_iris_centers(points)
                    gaze = normalized_gaze(points, (frame.shape[1], frame.shape[0]))
                    if iris_centers is not None:
                        for center in iris_centers:
                            pixel = (
                                round(center[0] * frame.shape[1]),
                                round(center[1] * frame.shape[0]),
                            )
                            cv2.circle(
                                frame,
                                pixel,
                                5,
                                (255, 0, 255),
                                thickness=2,
                                lineType=cv2.LINE_AA,
                            )
                    gaze_status = (
                        f"Gaze (normalized): gx={gaze[0]:.3f}, gy={gaze[1]:.3f}"
                        if gaze is not None
                        else "Gaze unavailable (open both eyes and face camera)"
                    )
                else:
                    status = "No face detected"
                    gaze_status = "Gaze unavailable"

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
                    gaze_status,
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 255) if faces and gaze is not None else (0, 0, 255),
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


def diagnosis_report(
    targets: list[tuple[int, int]],
    predicted_positions: list[tuple[float, float] | None],
    screen_size: tuple[int, int],
) -> tuple[list[str], float | None, float | None, float | None]:
    if len(targets) != len(predicted_positions):
        raise ValueError("Diagnosis needs one predicted position per target.")
    width, height = screen_size
    if width <= 0 or height <= 0:
        raise ValueError("Screen dimensions must be positive.")

    diagonal = math.hypot(width, height)
    rows = [
        f"{'Target':<14}{'Predicted':<16}{'Error(px)':<12}Error(% diag)"
    ]
    errors: list[float] = []
    x_biases: list[float] = []
    y_biases: list[float] = []
    for target, predicted in zip(targets, predicted_positions):
        if predicted is None:
            rows.append(f"{str(target):<14}{'N/A':<16}{'N/A':<12}N/A")
            continue
        error = math.dist(predicted, target)
        errors.append(error)
        x_biases.append(predicted[0] - target[0])
        y_biases.append(predicted[1] - target[1])
        rows.append(
            f"{str(target):<14}"
            f"{f'({predicted[0]:.0f}, {predicted[1]:.0f})':<16}"
            f"{error:<12.0f}{error / diagonal:.1%}"
        )

    mean_error = verification_mean_error(errors)
    x_bias = verification_mean_error(x_biases)
    y_bias = verification_mean_error(y_biases)
    rows.append("")
    rows.append(
        f"Mean error: {mean_error:.0f} px   "
        f"({mean_error / diagonal:.1%} of diagonal)"
        if mean_error is not None
        else "Mean error: N/A (no valid gaze samples)"
    )
    rows.append(
        f"X-axis bias: {x_bias:+.0f} px  "
        "(predicted-right of target on average)"
        if x_bias is not None
        else "X-axis bias: N/A"
    )
    rows.append(
        f"Y-axis bias: {y_bias:+.0f} px  "
        "(positive is below target; negative is above)"
        if y_bias is not None
        else "Y-axis bias: N/A"
    )
    return rows, mean_error, x_bias, y_bias


def run_diagnose(camera_index: int = 0) -> None:
    try:
        config_status = calibration_config_status()
    except OSError as error:
        raise RuntimeError(
            f"Cannot read calibration config {CALIBRATION_CONFIG}: {error}"
        ) from error
    if config_status == "missing":
        raise RuntimeError(
            "Cannot run --diagnose: calibration config is missing. "
            "Run --calibrate and complete cursor verification first."
        )
    try:
        config = json.loads(CALIBRATION_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read calibration config {CALIBRATION_CONFIG}: {error}"
        ) from error
    if not isinstance(config, dict):
        raise RuntimeError(
            f"Cannot run --diagnose: calibration config {CALIBRATION_CONFIG} "
            "is corrupt."
        )
    if config.get("verified") is not True:
        raise RuntimeError(
            "Cannot run --diagnose: calibration config is unverified. "
            "Run --calibrate and complete cursor verification first."
        )
    if config_status != "valid":
        raise RuntimeError(
            f"Cannot run --diagnose: calibration config is {config_status} "
            "or has invalid coefficient data."
        )
    x_coefficients = config["screen_x_coefficients"]
    y_coefficients = config["screen_y_coefficients"]

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
        pygame.init()
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Gaze Control Diagnosis")
        clock = pygame.time.Clock()
        title_font = pygame.font.Font(None, 42)
        body_font = pygame.font.Font(None, 27)
        small_font = pygame.font.Font(None, 21)
        width, height = screen.get_size()
    except Exception:
        capture.release()
        pygame.quit()
        raise

    targets = [
        (round(width * x), round(height * y))
        for y in (0.1, 0.5, 0.9)
        for x in (0.1, 0.5, 0.9)
    ]
    predicted_positions: list[tuple[float, float] | None] = []
    was_cancelled = False
    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            for target_index, target in enumerate(targets):
                target_samples: list[tuple[float, float]] = []
                phase = "countdown"
                phase_started = pygame.time.get_ticks() / 1000
                while phase != "done":
                    now = pygame.time.get_ticks() / 1000
                    phase_elapsed = now - phase_started
                    predicted: tuple[float, float] | None = None
                    frame_ok, frame = capture.read()
                    if not frame_ok:
                        raise RuntimeError("Failed to read a frame from the camera.")
                    result = face_mesh.process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    faces = result.multi_face_landmarks
                    if (
                        phase == "sample"
                        and faces
                    ):
                        points = [
                            (landmark.x, landmark.y)
                            for landmark in faces[0].landmark
                        ]
                        gaze = normalized_gaze(
                            points, (frame.shape[1], frame.shape[0])
                        )
                        if gaze is not None:
                            predicted = predict_screen_position(
                                gaze, x_coefficients, y_coefficients
                            )
                            if phase_elapsed < 2.0:
                                target_samples.append(predicted)

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN
                            and event.key in (pygame.K_ESCAPE, pygame.K_q)
                        ):
                            was_cancelled = True
                            phase = "done"

                    if phase == "countdown" and phase_elapsed >= 3.0:
                        phase = "sample"
                        phase_started = now
                    elif phase == "sample" and phase_elapsed >= 2.0:
                        predicted_positions.append(
                            (
                                sum(position[0] for position in target_samples)
                                / len(target_samples),
                                sum(position[1] for position in target_samples)
                                / len(target_samples),
                            )
                            if target_samples
                            else None
                        )
                        phase = "done"

                    screen.fill((12, 16, 26))
                    pygame.draw.circle(screen, (80, 190, 255), target, 16, width=2)
                    pygame.draw.circle(screen, (235, 245, 255), target, 4)
                    title = title_font.render(
                        f"Diagnosis target {target_index + 1}/9",
                        True,
                        (235, 239, 250),
                    )
                    screen.blit(title, title.get_rect(center=(width // 2, 48)))
                    if phase == "countdown":
                        number = countdown_number(phase_elapsed)
                        pulse = 1.0 + 0.12 * abs(math.sin(now * math.pi))
                        rendered_number = title_font.render(
                            str(number), True, (255, 245, 210)
                        )
                        rendered_number = pygame.transform.smoothscale(
                            rendered_number,
                            (
                                round(rendered_number.get_width() * pulse),
                                round(rendered_number.get_height() * pulse),
                            ),
                        )
                        screen.blit(
                            rendered_number,
                            rendered_number.get_rect(center=target),
                        )
                        instruction = (
                            "Look at the target; sampling begins after countdown."
                        )
                    else:
                        instruction = (
                            f"Sampling predicted cursor: "
                            f"{max(0.0, 2.0 - phase_elapsed):.1f}s"
                        )
                    screen.blit(
                        body_font.render(instruction, True, (195, 204, 220)),
                        body_font.render(instruction, True, (195, 204, 220))
                        .get_rect(center=(width // 2, height - 60)),
                    )
                    if predicted is not None:
                        cursor = (
                            min(max(round(predicted[0]), 0), width - 1),
                            min(max(round(predicted[1]), 0), height - 1),
                        )
                        pygame.draw.circle(
                            screen, (110, 255, 165), cursor, 10, width=2
                        )
                        pygame.draw.line(
                            screen,
                            (110, 255, 165),
                            (cursor[0] - 7, cursor[1]),
                            (cursor[0] + 7, cursor[1]),
                            2,
                        )
                        pygame.draw.line(
                            screen,
                            (110, 255, 165),
                            (cursor[0], cursor[1] - 7),
                            (cursor[0], cursor[1] + 7),
                            2,
                        )
                    pygame.display.flip()
                    clock.tick(30)
                    if was_cancelled:
                        break
                if was_cancelled:
                    break

            if not was_cancelled:
                report, _, _, _ = diagnosis_report(
                    targets, predicted_positions, (width, height)
                )
                print("\n".join(report))
                report_started = pygame.time.get_ticks() / 1000
                report_font = pygame.font.Font(None, 22)
                while pygame.time.get_ticks() / 1000 - report_started < 5.0:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN
                            and event.key in (pygame.K_ESCAPE, pygame.K_q)
                        ):
                            was_cancelled = True
                    if was_cancelled:
                        break
                    screen.fill((12, 16, 26))
                    report_title = title_font.render(
                        "Nine-target diagnosis report",
                        True,
                        (235, 239, 250),
                    )
                    screen.blit(
                        report_title,
                        report_title.get_rect(center=(width // 2, 48)),
                    )
                    line_spacing = min(
                        27, max(16, (height - 150) // max(1, len(report) - 1))
                    )
                    start_y = max(
                        75, (height - (len(report) - 1) * line_spacing) // 2
                    )
                    for line_index, line in enumerate(report):
                        rendered = report_font.render(
                            line, True, (220, 225, 235)
                        )
                        screen.blit(
                            rendered,
                            rendered.get_rect(
                                center=(
                                    width // 2,
                                    start_y + line_index * line_spacing,
                                )
                            ),
                        )
                    screen.blit(
                        small_font.render(
                            "Free-look starts next; press Q or Esc to exit.",
                            True,
                            (165, 180, 200),
                        ),
                        small_font.render(
                            "Free-look starts next; press Q or Esc to exit.",
                            True,
                            (165, 180, 200),
                        ).get_rect(center=(width // 2, height - 35)),
                    )
                    pygame.display.flip()
                    clock.tick(30)

            if not was_cancelled:
                free_look_started = pygame.time.get_ticks() / 1000
                while pygame.time.get_ticks() / 1000 - free_look_started < 20.0:
                    now = pygame.time.get_ticks() / 1000
                    frame_ok, frame = capture.read()
                    if not frame_ok:
                        raise RuntimeError("Failed to read a frame from the camera.")
                    result = face_mesh.process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    predicted: tuple[float, float] | None = None
                    if result.multi_face_landmarks:
                        points = [
                            (landmark.x, landmark.y)
                            for landmark in result.multi_face_landmarks[0].landmark
                        ]
                        gaze = normalized_gaze(
                            points, (frame.shape[1], frame.shape[0])
                        )
                        if gaze is not None:
                            predicted = predict_screen_position(
                                gaze, x_coefficients, y_coefficients
                            )
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN
                            and event.key in (pygame.K_ESCAPE, pygame.K_q)
                        ):
                            was_cancelled = True
                    screen.fill((12, 16, 26))
                    remaining = max(
                        0, 20 - int(pygame.time.get_ticks() / 1000 - free_look_started)
                    )
                    free_look_title = title_font.render(
                        f"Free-look — raw cursor — {remaining}s",
                        True,
                        (235, 239, 250),
                    )
                    screen.blit(
                        free_look_title,
                        free_look_title.get_rect(center=(width // 2, 48)),
                    )
                    if predicted is not None:
                        cursor = (
                            min(max(round(predicted[0]), 0), width - 1),
                            min(max(round(predicted[1]), 0), height - 1),
                        )
                        pygame.draw.circle(
                            screen, (110, 255, 165), cursor, 12, width=2
                        )
                        pygame.draw.line(
                            screen,
                            (110, 255, 165),
                            (cursor[0] - 8, cursor[1]),
                            (cursor[0] + 8, cursor[1]),
                            2,
                        )
                        pygame.draw.line(
                            screen,
                            (110, 255, 165),
                            (cursor[0], cursor[1] - 8),
                            (cursor[0], cursor[1] + 8),
                            2,
                        )
                    pygame.display.flip()
                    clock.tick(30)
                    if was_cancelled:
                        break
    finally:
        capture.release()
        pygame.quit()


def hold_error_screen(message: str) -> None:
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Gaze Control Error")
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    width, height = screen.get_size()
    lines = textwrap.wrap(message, width=75) or [message]
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN
                and event.key in (pygame.K_ESCAPE, pygame.K_q)
            ):
                running = False

        screen.fill((20, 12, 18))
        title = font.render("Gaze Control stopped after an error", True, (255, 145, 135))
        screen.blit(title, title.get_rect(center=(width // 2, height // 2 - 70)))
        for index, line in enumerate(lines[:5]):
            rendered = small_font.render(line, True, (230, 220, 220))
            screen.blit(
                rendered,
                rendered.get_rect(
                    center=(width // 2, height // 2 - 20 + index * 30)
                ),
            )
        action = small_font.render("Press Q or Esc to close.", True, (195, 204, 220))
        screen.blit(action, action.get_rect(center=(width // 2, height // 2 + 150)))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Gaze Control face, iris, calibration, and diagnosis tools."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index to open (default: 0).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run or repeat the full-screen nine-point calibration (Stage 3).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open the face/iris diagnostic preview without calibration.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run read-only nine-point and free-look gaze diagnosis.",
    )
    args = parser.parse_args()

    keep_error_window = not args.diagnose
    try:
        if args.diagnose:
            run_diagnose(args.camera_index)
        elif args.calibrate:
            run_calibration(args.camera_index)
        elif args.preview:
            run(args.camera_index)
        else:
            config_status = calibration_config_status()
            if config_status != "valid":
                keep_error_window = True
                print(
                    f"Calibration config is {config_status}; starting calibration.",
                    file=sys.stderr,
                )
                run_calibration(args.camera_index)
            else:
                run(args.camera_index)
    except Exception as error:
        traceback.print_exc()
        if keep_error_window:
            hold_error_screen(str(error))
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
