"""
UI Manager module.
Renders:
- Modern header toolbar with Clear button and Color palette
- Active tool/color indicators
- State badges (DRAWING, SELECTION, ERASING, PANNING, IDLE)
- Real-time FPS counter and gesture shortcut guides
- Interactive reticles/cursors tailored for each gesture mode
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    CLEAR_BTN_COLOR,
    CLEAR_BTN_HOVER,
    COLORS,
    DEFAULT_COLOR_NAME,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HEADER_HEIGHT,
    MODE_DRAWING,
    MODE_ERASING,
    MODE_IDLE,
    MODE_PANNING,
    MODE_SELECTION,
    UI_ACTIVE_BORDER,
    UI_BG_COLOR,
    UI_BORDER_COLOR,
    UI_TEXT_COLOR,
)


class UIButton:
    def __init__(self, name: str, rect: Tuple[int, int, int, int], color: Tuple[int, int, int], label: str = ""):
        self.name = name
        self.x1, self.y1, self.x2, self.y2 = rect
        self.color = color
        self.label = label or name

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class UIManager:
    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.width = width
        self.height = height
        self.header_height = HEADER_HEIGHT
        self.active_color_name = DEFAULT_COLOR_NAME
        self.buttons: List[UIButton] = []
        self._setup_buttons()

    def _setup_buttons(self):
        """Constructs layout bounding boxes for toolbar buttons."""
        self.buttons.clear()

        # Clear Button on Far Left
        clear_x1, clear_y1 = 20, 15
        clear_x2, clear_y2 = 150, self.header_height - 15
        self.buttons.append(UIButton("CLEAR", (clear_x1, clear_y1, clear_x2, clear_y2), CLEAR_BTN_COLOR, "CLEAR"))

        # Color Buttons
        palette_keys = list(COLORS.keys())
        num_colors = len(palette_keys)
        start_x = 180
        avail_width = 660  # Dedicated width for color swatches
        gap = 15
        btn_width = (avail_width - (num_colors - 1) * gap) // num_colors
        btn_y1 = 15
        btn_y2 = self.header_height - 15

        for i, color_name in enumerate(palette_keys):
            x1 = start_x + i * (btn_width + gap)
            x2 = x1 + btn_width
            self.buttons.append(UIButton(color_name, (x1, btn_y1, x2, btn_y2), COLORS[color_name], color_name))

    def check_interaction(self, point: Tuple[int, int]) -> Optional[str]:
        """Checks if a point (e.g. index tip) falls inside any toolbar button."""
        x, y = point
        if y > self.header_height:
            return None

        for btn in self.buttons:
            if btn.contains(x, y):
                return btn.name
        return None

    def render_header(self, frame: np.ndarray, hover_point: Optional[Tuple[int, int]] = None):
        """Draws the top toolbar containing buttons, palette, and borders."""
        # Header background with slight alpha blend for sleek glassmorphic effect
        header_overlay = frame.copy()
        cv2.rectangle(header_overlay, (0, 0), (self.width, self.header_height), UI_BG_COLOR, -1)
        cv2.addWeighted(header_overlay, 0.90, frame, 0.10, 0, frame)

        # Bottom accent divider line
        cv2.line(frame, (0, self.header_height), (self.width, self.header_height), UI_BORDER_COLOR, 2)

        # Draw Buttons
        for btn in self.buttons:
            is_hovered = hover_point and btn.contains(hover_point[0], hover_point[1])
            is_active = (btn.name == self.active_color_name)

            # Button background fill
            fill_color = btn.color
            if btn.name == "CLEAR":
                fill_color = CLEAR_BTN_HOVER if is_hovered else CLEAR_BTN_COLOR

            # Draw rounded/rectangular button
            cv2.rectangle(frame, (btn.x1, btn.y1), (btn.x2, btn.y2), fill_color, -1)

            # Border styling
            if is_active:
                cv2.rectangle(frame, (btn.x1 - 3, btn.y1 - 3), (btn.x2 + 3, btn.y2 + 3), UI_ACTIVE_BORDER, 3)
            elif is_hovered:
                cv2.rectangle(frame, (btn.x1 - 2, btn.y1 - 2), (btn.x2 + 2, btn.y2 + 2), (200, 200, 255), 2)
            else:
                cv2.rectangle(frame, (btn.x1, btn.y1), (btn.x2, btn.y2), UI_BORDER_COLOR, 1)

            # Text label
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.55 if btn.name != "CLEAR" else 0.65
            thickness = 1 if btn.name != "CLEAR" else 2
            text_size = cv2.getTextSize(btn.label, font, scale, thickness)[0]
            text_x = btn.x1 + (btn.x2 - btn.x1 - text_size[0]) // 2
            text_y = btn.y1 + (btn.y2 - btn.y1 + text_size[1]) // 2

            # Text color: dark text on light swatches, light text on dark swatches
            lum = (0.299 * fill_color[2] + 0.587 * fill_color[1] + 0.114 * fill_color[0])
            txt_color = (20, 20, 20) if lum > 160 else (250, 250, 250)
            cv2.putText(frame, btn.label, (text_x, text_y), font, scale, txt_color, thickness, cv2.LINE_AA)

    def render_hud(
        self,
        frame: np.ndarray,
        mode: str,
        fps: float,
        brush_size: int,
        pure_canvas: bool,
        show_skeleton: bool = True,
        fingers_summary: str = "",
    ):
        """Draws status badge, brush indicator, FPS, and bottom helper bar."""
        # Mode Badge Configuration
        mode_configs = {
            MODE_DRAWING: ("DRAWING (1 Finger)", (50, 220, 50)),
            MODE_SELECTION: ("SELECTION (2 Fingers)", (0, 200, 255)),
            MODE_ERASING: ("ERASER (4 Fingers)", (50, 50, 245)),
            MODE_PANNING: ("MOVING CANVAS (Open Palm)", (255, 140, 50)),
            MODE_IDLE: ("IDLE (Waiting for Hand)", (120, 120, 130)),
        }
        badge_text, badge_color = mode_configs.get(mode, ("UNKNOWN", (100, 100, 100)))

        # Mode Badge Pill (Top Right)
        badge_x2 = self.width - 20
        badge_w = 280
        badge_x1 = badge_x2 - badge_w
        badge_y1 = 15
        badge_y2 = 55
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (30, 30, 38), -1)
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_color, 2)
        cv2.circle(frame, (badge_x1 + 18, (badge_y1 + badge_y2) // 2), 6, badge_color, -1)
        cv2.putText(
            frame,
            badge_text,
            (badge_x1 + 32, (badge_y1 + badge_y2) // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            UI_TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

        # FPS, Skeleton & Fingers Info
        view_text = "Pure Canvas" if pure_canvas else "AR Feed"
        skel_text = "ON" if show_skeleton else "OFF"
        info_str = f"FPS: {int(fps)} | Brush: {brush_size}px | View: {view_text} | Skel[H]: {skel_text}"
        if fingers_summary:
            info_str += f" | {fingers_summary}"

        cv2.putText(
            frame,
            info_str,
            (badge_x1 - 60, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (210, 210, 220),
            1,
            cv2.LINE_AA,
        )

        # Bottom Shortcut Helper Bar
        guide_text = "1:Draw | 2:Select | 4:Erase | 5:Move Canvas || Keys: [C]lear [S]ave [V]iew [H]and Skeleton [+/-]Size [Q]uit"
        bar_y = self.height - 25
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.height - 35), (self.width, self.height), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(
            frame,
            guide_text,
            (20, bar_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 190, 200),
            1,
            cv2.LINE_AA,
        )

    def render_cursor(
        self,
        frame: np.ndarray,
        point: Tuple[int, int],
        mode: str,
        color: Tuple[int, int, int],
        brush_size: int,
        eraser_radius: int,
    ):
        """Draws context-aware visual feedback at the active hand position."""
        cx, cy = point

        if mode == MODE_DRAWING:
            # Drawing brush reticle
            cv2.circle(frame, (cx, cy), max(brush_size // 2, 4), color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), max(brush_size // 2, 4) + 4, (255, 255, 255), 1, cv2.LINE_AA)

        elif mode == MODE_SELECTION:
            # Sleek selection pointer
            cv2.circle(frame, (cx, cy), 12, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

        elif mode == MODE_ERASING:
            # Large eraser circle showing exact erasure footprint
            cv2.circle(frame, (cx, cy), eraser_radius, (50, 50, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                "ERASER",
                (cx - 25, cy - eraser_radius - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (50, 50, 255),
                1,
                cv2.LINE_AA,
            )

        elif mode == MODE_PANNING:
            # 4-Directional movement crosshair
            length = 25
            cv2.arrowedLine(frame, (cx, cy), (cx + length, cy), (255, 140, 50), 2, cv2.LINE_AA)
            cv2.arrowedLine(frame, (cx, cy), (cx - length, cy), (255, 140, 50), 2, cv2.LINE_AA)
            cv2.arrowedLine(frame, (cx, cy), (cx, cy - length), (255, 140, 50), 2, cv2.LINE_AA)
            cv2.arrowedLine(frame, (cx, cy), (cx, cy + length), (255, 140, 50), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
