"""
Canvas Manager module.
Handles:
- Dynamic drawing canvas layer with anti-aliased line rendering
- Four-finger motion-based eraser with continuous stroke interpolation
- Whole-palm affine translation (moving drawn content across the screen)
- Bitwise and alpha blending between the camera feed and the virtual canvas
- Canvas clearing and snapshot exports
"""

import os
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import (
    DEFAULT_BRUSH_THICKNESS,
    ERASER_RADIUS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MIN_MOVE_DISTANCE,
)


class CanvasManager:
    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.width = width
        self.height = height

        # Drawing layer: black (0,0,0) represents empty/transparent
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Stroke tracking state
        self.prev_draw_pt: Optional[Tuple[int, int]] = None
        self.prev_erase_pt: Optional[Tuple[int, int]] = None
        self.prev_pan_pt: Optional[Tuple[int, int]] = None

        # Brush settings
        self.brush_thickness = DEFAULT_BRUSH_THICKNESS
        self.eraser_radius = ERASER_RADIUS

    def draw_stroke(self, point: Tuple[int, int], color: Tuple[int, int, int]):
        """
        Draws continuous lines connecting sequential points in Drawing Mode.
        """
        curr_x, curr_y = point

        if self.prev_draw_pt is None:
            self.prev_draw_pt = (curr_x, curr_y)
            return

        # Avoid redundant draws if finger barely moved
        dist = np.hypot(curr_x - self.prev_draw_pt[0], curr_y - self.prev_draw_pt[1])
        if dist < MIN_MOVE_DISTANCE:
            return

        # Draw anti-aliased line between previous and current point
        cv2.line(
            self.canvas,
            self.prev_draw_pt,
            (curr_x, curr_y),
            color,
            self.brush_thickness,
            lineType=cv2.LINE_AA,
        )
        # Add smooth end caps
        cv2.circle(
            self.canvas,
            (curr_x, curr_y),
            self.brush_thickness // 2,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )

        self.prev_draw_pt = (curr_x, curr_y)

    def erase_at(self, point: Tuple[int, int]):
        """
        Erases canvas content along the movement path of the 4 fingers.
        Uses continuous interpolation to avoid gaps during fast hand motions.
        """
        curr_x, curr_y = point

        if self.prev_erase_pt is None:
            self.prev_erase_pt = (curr_x, curr_y)

        # Erase continuous corridor between previous and current position
        cv2.line(
            self.canvas,
            self.prev_erase_pt,
            (curr_x, curr_y),
            (0, 0, 0),
            thickness=self.eraser_radius * 2,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            self.canvas,
            (curr_x, curr_y),
            self.eraser_radius,
            (0, 0, 0),
            -1,
            lineType=cv2.LINE_AA,
        )

        self.prev_erase_pt = (curr_x, curr_y)

    def pan_canvas(self, palm_center: Tuple[int, int]):
        """
        Moves the entire drawn content across the screen based on whole palm movement delta (dx, dy).
        """
        curr_x, curr_y = palm_center

        if self.prev_pan_pt is None:
            self.prev_pan_pt = (curr_x, curr_y)
            return

        dx = curr_x - self.prev_pan_pt[0]
        dy = curr_y - self.prev_pan_pt[1]

        # Only apply translation if motion is significant
        if abs(dx) > 1 or abs(dy) > 1:
            # 2D Affine transformation matrix
            translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            self.canvas = cv2.warpAffine(
                self.canvas,
                translation_matrix,
                (self.width, self.height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        self.prev_pan_pt = (curr_x, curr_y)

    def reset_stroke(self):
        """Resets sequential tracking pointers when lifting fingers or changing modes."""
        self.prev_draw_pt = None
        self.prev_erase_pt = None
        self.prev_pan_pt = None

    def clear(self):
        """Wipes the entire canvas layer."""
        self.canvas[:] = 0
        self.reset_stroke()

    def blend(self, frame: np.ndarray, pure_canvas: bool = False) -> np.ndarray:
        """
        Blends the canvas layer with the webcam frame.
        - If pure_canvas is True: Returns a dark digital drawing board with the strokes.
        - If pure_canvas is False: Seamlessly overlays strokes onto the camera feed using bitwise masking.
        """
        if pure_canvas:
            # Clean dark slate digital drawing board
            board = np.full_like(self.canvas, fill_value=25, dtype=np.uint8)
            # Combine board with colored strokes
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            board_bg = cv2.bitwise_and(board, board, mask=cv2.bitwise_not(mask))
            return cv2.add(board_bg, self.canvas)

        # Standard AR Overlay:
        # 1. Convert canvas to grayscale
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        # 2. Threshold to find pixels containing drawings
        _, mask_inv = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        # 3. Mask out drawn regions from the camera feed
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        # 4. Add the colored strokes into the masked region
        blended = cv2.add(frame_bg, self.canvas)
        return blended

    def save_snapshot(self, frame_to_save: np.ndarray, output_dir: str = "snapshots") -> str:
        """Saves current canvas/frame view to a timestamped PNG file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"canvas_snapshot_{timestamp}.png")
        cv2.imwrite(filename, frame_to_save)
        return filename
