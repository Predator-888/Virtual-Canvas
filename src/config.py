"""
Configuration and constants for the Virtual Air Canvas application.
"""

from typing import Dict, Tuple

# Window and Display Configuration
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_TITLE = "Virtual Air Canvas - Next-Gen Gesture Studio"
HEADER_HEIGHT = 100

# Color Palette (BGR Format for OpenCV)
COLORS: Dict[str, Tuple[int, int, int]] = {
    "BLUE": (255, 120, 0),       # Vibrant Sky/Electric Blue
    "GREEN": (50, 225, 50),      # Vivid Emerald Green
    "RED": (50, 50, 255),        # Bright Crimson Red
    "YELLOW": (0, 220, 255),     # Radiant Yellow
    "PURPLE": (220, 60, 180),    # Electric Violet
    "ORANGE": (0, 140, 255),     # Deep Orange
}

# UI Theme Colors
UI_BG_COLOR = (24, 24, 30)           # Dark charcoal header background
UI_BORDER_COLOR = (60, 60, 75)       # Subtle border
UI_TEXT_COLOR = (240, 240, 245)      # Near white
UI_ACTIVE_BORDER = (255, 255, 255)   # White highlight for active item
CLEAR_BTN_COLOR = (45, 45, 60)       # Dark slate for Clear button
CLEAR_BTN_HOVER = (70, 70, 95)

# Drawing & Brush Settings
DEFAULT_COLOR_NAME = "BLUE"
DEFAULT_BRUSH_THICKNESS = 7
MIN_BRUSH_THICKNESS = 2
MAX_BRUSH_THICKNESS = 40
ERASER_RADIUS = 45

# Gesture / Mode Definitions
MODE_DRAWING = "DRAWING"       # 1 finger (Index)
MODE_SELECTION = "SELECTION"   # 2 fingers (Index + Middle)
MODE_ERASING = "ERASING"       # 4 fingers (Index + Middle + Ring + Pinky)
MODE_PANNING = "PANNING"       # 5 fingers (Open Palm)
MODE_IDLE = "IDLE"             # Fist or unclassified

# MediaPipe Confidence Parameters (Tuned for reliable real-time webcam tracking)
DETECTION_CONFIDENCE = 0.50
TRACKING_CONFIDENCE = 0.50
PRESENCE_CONFIDENCE = 0.45

# Stability & Grace Frames (Prevents tracking dropout on fast motion)
HAND_LOST_GRACE_FRAMES = 5         # Frames to retain previous hand state before dropping
GESTURE_CONFIRMATION_FRAMES = 2    # Consecutive frames needed to confirm a mode switch

# Jitter & Smoothing
SMOOTHING_ALPHA = 0.55             # Exponential moving average factor (0.0 to 1.0)
MIN_MOVE_DISTANCE = 2              # Minimum movement in pixels to record point

