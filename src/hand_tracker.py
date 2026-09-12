"""
MediaPipe Hand Tracking and Gesture Recognition Module.
Engineered for ultra-smooth real-time tracking with:
- Temporal Video Tracking Mode (RunningMode.VIDEO)
- Hand Persistence Grace Buffer (prevents dropout on rapid motion or blurs)
- Geometric 3D-aware finger extension detection
- Temporal Gesture Stabilization / Debouncing
"""

import os
import time
import urllib.request
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.config import (
    DETECTION_CONFIDENCE,
    GESTURE_CONFIRMATION_FRAMES,
    HAND_LOST_GRACE_FRAMES,
    MODE_DRAWING,
    MODE_ERASING,
    MODE_IDLE,
    MODE_PANNING,
    MODE_SELECTION,
    PRESENCE_CONFIDENCE,
    SMOOTHING_ALPHA,
    TRACKING_CONFIDENCE,
)

# Hand skeleton connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = DETECTION_CONFIDENCE,
        tracking_confidence: float = TRACKING_CONFIDENCE,
    ):
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Filtered / Smoothed Landmark Coordinates
        self.smoothed_landmarks: Optional[List[Tuple[int, int]]] = None

        # Stability & Persistence State
        self.consecutive_lost_frames = 0
        self.current_mode = MODE_IDLE
        self.candidate_mode = MODE_IDLE
        self.candidate_mode_count = 0

        # Monotonic Video Timestamp (ms)
        self.last_timestamp_ms = 0
        self.start_time = time.time()

        # Finger landmark indices
        self.tip_ids = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
        self.dip_ids = [3, 7, 11, 15, 19]
        self.pip_ids = [2, 6, 10, 14, 18]
        self.mcp_ids = [1, 5, 9, 13, 17]

        # Determine which MediaPipe API is available
        self.use_tasks_api = not hasattr(mp, "solutions")

        if self.use_tasks_api:
            self._init_tasks_api()
        else:
            self._init_solutions_api()

    def _ensure_model_exists(self) -> str:
        """Ensures the hand landmarker model file is available locally."""
        model_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(model_dir, ".."))
        local_model_path = os.path.join(project_root, MODEL_FILENAME)

        if not os.path.exists(local_model_path):
            print(f"📦 Downloading MediaPipe hand landmarker model to {local_model_path}...")
            urllib.request.urlretrieve(MODEL_URL, local_model_path)
            print("✅ Model downloaded successfully.")

        return local_model_path

    def _init_tasks_api(self):
        """Initializes modern MediaPipe Tasks Vision HandLandmarker in VIDEO tracking mode."""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = self._ensure_model_exists()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.detection_confidence,
            min_hand_presence_confidence=PRESENCE_CONFIDENCE,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _init_solutions_api(self):
        """Initializes legacy MediaPipe Solutions Hands."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

    def _get_next_timestamp_ms(self) -> int:
        """Generates strictly increasing monotonic timestamps for RunningMode.VIDEO."""
        current_ms = int((time.time() - self.start_time) * 1000)
        if current_ms <= self.last_timestamp_ms:
            current_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = current_ms
        return current_ms

    def find_hands(self, frame: np.ndarray, draw: bool = False) -> Tuple[np.ndarray, Optional[any]]:
        """
        Process the frame to detect hands using video tracking.
        Returns the annotated frame and the results object.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = self._get_next_timestamp_ms()
            results = self.detector.detect_for_video(mp_image, timestamp_ms)
            landmarks = self.extract_landmarks(frame, results)
            if draw and landmarks:
                self.draw_skeleton(frame, landmarks)
            return frame, results
        else:
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            landmarks = self.extract_landmarks(frame, results)
            if draw and landmarks:
                self.draw_skeleton(frame, landmarks)
            return frame, results

    def extract_landmarks(self, frame: np.ndarray, results: any) -> Optional[List[Tuple[int, int]]]:
        """
        Extracts pixel coordinates (x, y) for all 21 hand landmarks.
        Includes a persistence grace buffer so 1-4 dropped frames do not interrupt drawing.
        """
        h, w, _ = frame.shape
        raw_landmarks: List[Tuple[int, int]] = []
        has_detection = False

        if self.use_tasks_api:
            if results and results.hand_landmarks and len(results.hand_landmarks) > 0:
                hand = results.hand_landmarks[0]
                for lm in hand:
                    raw_landmarks.append((int(lm.x * w), int(lm.y * h)))
                has_detection = True
        else:
            if results and results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                hand = results.multi_hand_landmarks[0]
                for lm in hand.landmark:
                    raw_landmarks.append((int(lm.x * w), int(lm.y * h)))
                has_detection = True

        if has_detection:
            self.consecutive_lost_frames = 0
            # Apply exponential smoothing
            if self.smoothed_landmarks is None:
                self.smoothed_landmarks = raw_landmarks
            else:
                smoothed = []
                for (curr_x, curr_y), (prev_x, prev_y) in zip(raw_landmarks, self.smoothed_landmarks):
                    sx = int(SMOOTHING_ALPHA * curr_x + (1 - SMOOTHING_ALPHA) * prev_x)
                    sy = int(SMOOTHING_ALPHA * curr_y + (1 - SMOOTHING_ALPHA) * prev_y)
                    smoothed.append((sx, sy))
                self.smoothed_landmarks = smoothed
            return self.smoothed_landmarks

        # No detection in current frame -> Check Grace Buffer
        self.consecutive_lost_frames += 1
        if self.consecutive_lost_frames <= HAND_LOST_GRACE_FRAMES and self.smoothed_landmarks is not None:
            # Retain previous position through temporary dropout
            return self.smoothed_landmarks

        # Truly lost after exceeding grace period
        self.smoothed_landmarks = None
        self.current_mode = MODE_IDLE
        self.candidate_mode = MODE_IDLE
        self.candidate_mode_count = 0
        return None

    def draw_skeleton(self, frame: np.ndarray, landmarks: List[Tuple[int, int]]):
        """Draws aesthetic cyber hand connections and joints on the frame."""
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, landmarks[start_idx], landmarks[end_idx], (0, 220, 255), 2, cv2.LINE_AA)
        for i, pt in enumerate(landmarks):
            # Highlight fingertips
            if i in self.tip_ids:
                cv2.circle(frame, pt, 6, (0, 255, 120), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def get_fingers_up(self, landmarks: List[Tuple[int, int]]) -> List[bool]:
        """
        Determines which of the 5 fingers are extended using geometric distance ratios
        and joint angles that remain reliable across hand tilts and wrist angles.
        Returns [thumb, index, middle, ring, pinky] as booleans.
        """
        fingers = []
        wrist = np.array(landmarks[0])

        # 1. Thumb detection:
        # Measure distance from thumb tip to pinky MCP and thumb IP to pinky MCP
        thumb_tip = np.array(landmarks[4])
        thumb_ip = np.array(landmarks[3])
        pinky_mcp = np.array(landmarks[17])
        index_mcp = np.array(landmarks[5])

        dist_tip_to_pinky = np.linalg.norm(thumb_tip - pinky_mcp)
        dist_ip_to_pinky = np.linalg.norm(thumb_ip - pinky_mcp)
        dist_tip_to_index = np.linalg.norm(thumb_tip - index_mcp)

        # Thumb is open if extended outwards away from the palm
        is_thumb_up = bool(dist_tip_to_pinky > dist_ip_to_pinky * 1.10 and dist_tip_to_index > 35)
        fingers.append(is_thumb_up)

        # 2. Four Fingers (Index, Middle, Ring, Pinky):
        # A finger is extended when its tip is stretched away from the wrist.
        # When curled into a palm, the tip curls inward, making dist(tip, wrist) <= dist(pip, wrist).
        for i in range(1, 5):
            tip = np.array(landmarks[self.tip_ids[i]])
            pip = np.array(landmarks[self.pip_ids[i]])
            mcp = np.array(landmarks[self.mcp_ids[i]])

            dist_tip_wrist = np.linalg.norm(tip - wrist)
            dist_pip_wrist = np.linalg.norm(pip - wrist)
            dist_tip_mcp = np.linalg.norm(tip - mcp)
            dist_pip_mcp = np.linalg.norm(pip - mcp)

            # Combined ratio test: tip is farther from wrist than PIP joint, and tip is farther from MCP than PIP
            is_extended = (dist_tip_wrist > dist_pip_wrist * 1.08) and (dist_tip_mcp > dist_pip_mcp * 1.10)
            fingers.append(bool(is_extended))

        return fingers

    def classify_gesture(self, fingers: List[bool]) -> str:
        """
        Classifies the hand posture into a system mode with temporal stabilization (debouncing):
        - 1 Finger (Index): DRAWING
        - 2 Fingers (Index + Middle): SELECTION
        - 4 Fingers (Index + Middle + Ring + Pinky, thumb tucked): ERASING
        - 5 Fingers (Open Palm / all 5 extended): PANNING
        - Other: IDLE
        """
        thumb, index, middle, ring, pinky = fingers
        four_fingers = index and middle and ring and pinky

        if four_fingers:
            raw_mode = MODE_PANNING if thumb else MODE_ERASING
        elif index and middle and not ring and not pinky:
            raw_mode = MODE_SELECTION
        elif index and not middle and not ring and not pinky:
            raw_mode = MODE_DRAWING
        else:
            raw_mode = MODE_IDLE

        if not hasattr(self, "current_mode"):
            self.current_mode = MODE_IDLE
            self.candidate_mode = MODE_IDLE
            self.candidate_mode_count = 0

        # Temporal Gesture Debouncing / Smoothing:
        # Mode changes require consecutive frame confirmation to avoid single-frame flickers
        if raw_mode == self.current_mode:
            self.candidate_mode = raw_mode
            self.candidate_mode_count = 0
            return self.current_mode

        if raw_mode == self.candidate_mode:
            self.candidate_mode_count += 1
            if self.candidate_mode_count >= GESTURE_CONFIRMATION_FRAMES:
                self.current_mode = raw_mode
                self.candidate_mode_count = 0
        else:
            self.candidate_mode = raw_mode
            self.candidate_mode_count = 1

        return self.current_mode

    def get_palm_center(self, landmarks: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Calculates the centroid of the palm (wrist + MCP bases).
        """
        palm_indices = [0, 5, 9, 13, 17]
        avg_x = int(sum(landmarks[i][0] for i in palm_indices) / len(palm_indices))
        avg_y = int(sum(landmarks[i][1] for i in palm_indices) / len(palm_indices))
        return avg_x, avg_y

    def close(self):
        """Release MediaPipe resources."""
        if self.use_tasks_api and hasattr(self, "detector"):
            self.detector.close()
        elif hasattr(self, "hands"):
            self.hands.close()
