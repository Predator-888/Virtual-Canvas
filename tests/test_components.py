"""
Unit and Component Smoke Tests for Virtual Air Canvas.
Verifies CanvasManager, UIManager, and Gesture Classification logic.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HEADER_HEIGHT,
    MODE_DRAWING,
    MODE_ERASING,
    MODE_IDLE,
    MODE_PANNING,
    MODE_SELECTION,
)
from src.canvas import CanvasManager
from src.ui import UIManager
from src.hand_tracker import HandTracker


class TestVirtualCanvasComponents(unittest.TestCase):
    def setUp(self):
        self.dummy_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.canvas_mgr = CanvasManager(FRAME_WIDTH, FRAME_HEIGHT)
        self.ui_mgr = UIManager(FRAME_WIDTH, FRAME_HEIGHT)

    def test_canvas_drawing_and_clearing(self):
        # Initial canvas should be all zeros
        self.assertEqual(np.sum(self.canvas_mgr.canvas), 0)

        # Draw stroke
        self.canvas_mgr.draw_stroke((200, 200), (0, 255, 0))
        self.canvas_mgr.draw_stroke((250, 250), (0, 255, 0))

        # Canvas should now have non-zero pixels
        drawn_pixels = np.sum(self.canvas_mgr.canvas > 0)
        self.assertGreater(drawn_pixels, 0, "Canvas should have drawn pixels")

        # Clear canvas
        self.canvas_mgr.clear()
        self.assertEqual(np.sum(self.canvas_mgr.canvas), 0, "Canvas should be completely cleared")

    def test_canvas_erasing(self):
        # Draw something first
        self.canvas_mgr.draw_stroke((300, 300), (255, 0, 0))
        self.canvas_mgr.draw_stroke((350, 350), (255, 0, 0))
        drawn_before = np.sum(self.canvas_mgr.canvas > 0)
        self.assertGreater(drawn_before, 0)

        # Erase at the drawn location
        self.canvas_mgr.erase_at((325, 325))
        drawn_after = np.sum(self.canvas_mgr.canvas > 0)
        self.assertLess(drawn_after, drawn_before, "Drawn pixels should decrease after erase")

    def test_canvas_panning(self):
        # Draw a shape at (400, 400)
        self.canvas_mgr.draw_stroke((400, 400), (0, 0, 255))
        self.canvas_mgr.draw_stroke((420, 420), (0, 0, 255))

        # Apply pan delta of (+50, +50)
        self.canvas_mgr.pan_canvas((200, 200))
        self.canvas_mgr.pan_canvas((250, 250))  # dx=+50, dy=+50

        # Verify affine warp executed without error and canvas still has content
        self.assertGreater(np.sum(self.canvas_mgr.canvas > 0), 0)

    def test_blending(self):
        # Test AR blending
        blended_ar = self.canvas_mgr.blend(self.dummy_frame, pure_canvas=False)
        self.assertEqual(blended_ar.shape, (FRAME_HEIGHT, FRAME_WIDTH, 3))

        # Test Pure canvas blending
        blended_pure = self.canvas_mgr.blend(self.dummy_frame, pure_canvas=True)
        self.assertEqual(blended_pure.shape, (FRAME_HEIGHT, FRAME_WIDTH, 3))

    def test_ui_button_interactions(self):
        # Click on Clear button area (around x=50, y=50)
        btn = self.ui_mgr.check_interaction((50, 50))
        self.assertEqual(btn, "CLEAR")

        # Point below header should return None
        below_header = self.ui_mgr.check_interaction((50, HEADER_HEIGHT + 50))
        self.assertIsNone(below_header)

    def test_ui_rendering_smoke(self):
        frame = self.dummy_frame.copy()
        # Should render without throwing exceptions
        self.ui_mgr.render_header(frame, hover_point=(50, 50))
        self.ui_mgr.render_hud(frame, MODE_DRAWING, fps=30.0, brush_size=7, pure_canvas=False)
        self.ui_mgr.render_cursor(frame, (300, 300), MODE_DRAWING, (255, 0, 0), brush_size=7, eraser_radius=45)
        self.assertGreater(np.sum(frame > 0), 0)

    def test_gesture_classification(self):
        # Test dummy HandTracker classification logic without requiring camera
        tracker = HandTracker.__new__(HandTracker)
        tracker.current_mode = MODE_IDLE
        tracker.candidate_mode = MODE_IDLE
        tracker.candidate_mode_count = 0

        # Helper to simulate consecutive frames for debouncing
        def classify_stable(fingers):
            tracker.classify_gesture(fingers)
            return tracker.classify_gesture(fingers)

        # 1 Finger (Index): [Thumb, Index, Middle, Ring, Pinky] -> DRAWING
        self.assertEqual(classify_stable([False, True, False, False, False]), MODE_DRAWING)

        # 2 Fingers (Index + Middle) -> SELECTION
        self.assertEqual(classify_stable([False, True, True, False, False]), MODE_SELECTION)

        # 4 Fingers (Four fingers up, thumb curled) -> ERASER
        self.assertEqual(classify_stable([False, True, True, True, True]), MODE_ERASING)

        # 5 Fingers (All extended) -> PANNING
        self.assertEqual(classify_stable([True, True, True, True, True]), MODE_PANNING)

        # Fist / Other -> IDLE
        self.assertEqual(classify_stable([False, False, False, False, False]), MODE_IDLE)

    def test_hand_tracker_pipeline(self):
        # End-to-end test of HandTracker instantiation and frame processing
        tracker = HandTracker()
        frame, results = tracker.find_hands(self.dummy_frame, draw=False)
        landmarks = tracker.extract_landmarks(frame, results)
        # On a blank frame, no hand should be detected
        self.assertIsNone(landmarks)
        tracker.close()


if __name__ == "__main__":
    unittest.main()
