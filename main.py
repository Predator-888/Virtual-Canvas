"""
Virtual Air Canvas - Main Application Entry Point.
Integrates webcam video stream, MediaPipe hand tracking, dynamic canvas rendering,
and touchless spatial gesture controls.
"""

import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.canvas import CanvasManager
from src.config import (
    COLORS,
    DEFAULT_BRUSH_THICKNESS,
    DEFAULT_COLOR_NAME,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HEADER_HEIGHT,
    MAX_BRUSH_THICKNESS,
    MIN_BRUSH_THICKNESS,
    MODE_DRAWING,
    MODE_ERASING,
    MODE_IDLE,
    MODE_PANNING,
    MODE_SELECTION,
    WINDOW_TITLE,
)
from src.hand_tracker import HandTracker
from src.ui import UIManager


def main():
    print("=" * 60)
    print(" 🎨 Starting Virtual Air Canvas...")
    print("=" * 60)

    # Initialize Modules
    tracker = HandTracker(max_num_hands=1)
    canvas_manager = CanvasManager(width=FRAME_WIDTH, height=FRAME_HEIGHT)
    ui_manager = UIManager(width=FRAME_WIDTH, height=FRAME_HEIGHT)

    # Open Video Capture with DirectShow on Windows for zero-latency streaming
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\n❌ Error: Could not access the webcam (device 0).")
        print("Please ensure your webcam is connected and not in use by another application.")
        return

    # Configure Camera Properties: Disable driver buffering lag and lock resolution
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Create Named Window
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, FRAME_WIDTH, FRAME_HEIGHT)

    # Runtime State
    active_color_name = DEFAULT_COLOR_NAME
    active_color = COLORS[active_color_name]
    pure_canvas_mode = False
    show_skeleton = True
    prev_time = time.time()
    fps = 30.0

    # Debounce for UI Button Interactions
    last_button_click_time = 0.0
    BUTTON_DEBOUNCE_DELAY = 0.35  # seconds

    # Notification message system (e.g. "Saved Snapshot!")
    status_msg = ""
    status_msg_expiry = 0.0

    print("\n✅ Virtual Air Canvas is Running!")
    print("--------------------------------------------------")
    print(" Gesture Controls:")
    print("   👉 1 Finger  (Index):           DRAW in active color")
    print("   ✌️ 2 Fingers (Index + Middle):   SELECTION / HOVER (Select colors/clear)")
    print("   🖐️ 4 Fingers (Four Fingers):     ERASE content around hand movement")
    print("   ✋ Open Palm (All 5 Fingers):    MOVE / PAN canvas content across screen")
    print(" Keyboard Shortcuts:")
    print("   [C] Clear Canvas | [S] Save Snapshot | [V] Toggle Pure Canvas")
    print("   [H] Toggle Hand Skeleton | [+/-] Brush Size | [Q] or [ESC] Quit")
    print("--------------------------------------------------\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Warning: Failed to grab frame from camera.")
                break

            # 1. Mirror horizontally for natural, intuitive reflection
            frame = cv2.flip(frame, 1)

            # Ensure resolution matches configured dimensions
            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # 2. Calculate Real-Time FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # 3. Hand Tracking & Gesture Processing
            frame, results = tracker.find_hands(frame, draw=False)
            landmarks = tracker.extract_landmarks(frame, results)

            mode = MODE_IDLE
            active_cursor_pt: Optional[Tuple[int, int]] = None
            fingers_summary = ""

            if landmarks is not None:
                fingers = tracker.get_fingers_up(landmarks)
                mode = tracker.classify_gesture(fingers)

                # Compute readable finger summary for user HUD
                num_up = sum(fingers)
                fingers_summary = f"Fingers: {num_up}"

                # Key Landmark Coordinates
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                pinky_tip = landmarks[20]
                palm_center = tracker.get_palm_center(landmarks)

                # 4-Finger Centroid for the Movement Eraser
                four_finger_centroid = (
                    (index_tip[0] + middle_tip[0] + ring_tip[0] + pinky_tip[0]) // 4,
                    (index_tip[1] + middle_tip[1] + ring_tip[1] + pinky_tip[1]) // 4,
                )

                # Header Guard: If index finger is inside toolbar header, prevent drawing
                if index_tip[1] <= HEADER_HEIGHT and mode == MODE_DRAWING:
                    mode = MODE_SELECTION

                # Execute State Actions
                if mode == MODE_DRAWING:
                    active_cursor_pt = index_tip
                    canvas_manager.draw_stroke(index_tip, active_color)

                elif mode == MODE_SELECTION:
                    active_cursor_pt = index_tip
                    canvas_manager.reset_stroke()

                    # Check for button click in toolbar
                    if index_tip[1] <= HEADER_HEIGHT:
                        clicked_btn = ui_manager.check_interaction(index_tip)
                        if clicked_btn and (curr_time - last_button_click_time > BUTTON_DEBOUNCE_DELAY):
                            last_button_click_time = curr_time
                            if clicked_btn == "CLEAR":
                                canvas_manager.clear()
                                status_msg = "Canvas Cleared!"
                                status_msg_expiry = curr_time + 1.5
                            elif clicked_btn in COLORS:
                                active_color_name = clicked_btn
                                active_color = COLORS[active_color_name]
                                ui_manager.active_color_name = active_color_name
                                status_msg = f"Color: {active_color_name}"
                                status_msg_expiry = curr_time + 1.2

                elif mode == MODE_ERASING:
                    active_cursor_pt = four_finger_centroid
                    canvas_manager.reset_stroke()
                    canvas_manager.erase_at(four_finger_centroid)

                elif mode == MODE_PANNING:
                    active_cursor_pt = palm_center
                    canvas_manager.reset_stroke()
                    canvas_manager.pan_canvas(palm_center)

                else:  # MODE_IDLE
                    canvas_manager.reset_stroke()

            else:
                # Hand out of frame: reset stroke continuity
                canvas_manager.reset_stroke()

            # 4. Blend Drawing Layer with Camera Feed
            display_frame = canvas_manager.blend(frame, pure_canvas=pure_canvas_mode)

            # 5. Draw Hand Skeleton (if enabled and hand is tracked)
            if show_skeleton and landmarks is not None and not pure_canvas_mode:
                tracker.draw_skeleton(display_frame, landmarks)

            # 6. Render Modern UI Elements
            hover_pt = active_cursor_pt if (landmarks and active_cursor_pt and active_cursor_pt[1] <= HEADER_HEIGHT) else None
            ui_manager.render_header(display_frame, hover_point=hover_pt)
            ui_manager.render_hud(
                display_frame,
                mode=mode,
                fps=fps,
                brush_size=canvas_manager.brush_thickness,
                pure_canvas=pure_canvas_mode,
                show_skeleton=show_skeleton,
                fingers_summary=fingers_summary,
            )

            # 7. Render Context-Aware Dynamic Cursor
            if active_cursor_pt is not None:
                ui_manager.render_cursor(
                    frame=display_frame,
                    point=active_cursor_pt,
                    mode=mode,
                    color=active_color,
                    brush_size=canvas_manager.brush_thickness,
                    eraser_radius=canvas_manager.eraser_radius,
                )

            # 8. Render Transient Status Notification (if active)
            if curr_time < status_msg_expiry and status_msg:
                msg_size = cv2.getTextSize(status_msg, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
                msg_x = (FRAME_WIDTH - msg_size[0]) // 2
                msg_y = HEADER_HEIGHT + 45
                cv2.rectangle(
                    display_frame,
                    (msg_x - 15, msg_y - msg_size[1] - 10),
                    (msg_x + msg_size[0] + 15, msg_y + 10),
                    (20, 20, 25),
                    -1,
                )
                cv2.rectangle(
                    display_frame,
                    (msg_x - 15, msg_y - msg_size[1] - 10),
                    (msg_x + msg_size[0] + 15, msg_y + 10),
                    (0, 220, 255),
                    1,
                )
                cv2.putText(
                    display_frame,
                    status_msg,
                    (msg_x, msg_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # 9. Render to Window
            cv2.imshow(WINDOW_TITLE, display_frame)

            # 10. Handle Keyboard Shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC to quit
                print("\n🛑 Exiting Virtual Air Canvas...")
                break
            elif key in (ord("c"), ord("C")):    # Clear
                canvas_manager.clear()
                status_msg = "Canvas Cleared!"
                status_msg_expiry = curr_time + 1.5
            elif key in (ord("s"), ord("S")):    # Save snapshot
                filepath = canvas_manager.save_snapshot(display_frame)
                status_msg = "Snapshot Saved!"
                status_msg_expiry = curr_time + 2.0
                print(f"📸 Snapshot saved to: {filepath}")
            elif key in (ord("v"), ord("V")):    # Toggle pure canvas
                pure_canvas_mode = not pure_canvas_mode
                status_msg = "Pure Canvas Mode" if pure_canvas_mode else "AR Camera Mode"
                status_msg_expiry = curr_time + 1.5
            elif key in (ord("h"), ord("H")):    # Toggle hand skeleton
                show_skeleton = not show_skeleton
                status_msg = f"Skeleton: {'ON' if show_skeleton else 'OFF'}"
                status_msg_expiry = curr_time + 1.5
            elif key in (ord("+"), ord("=")):    # Increase brush size
                canvas_manager.brush_thickness = min(canvas_manager.brush_thickness + 2, MAX_BRUSH_THICKNESS)
            elif key in (ord("-"), ord("_")):    # Decrease brush size
                canvas_manager.brush_thickness = max(canvas_manager.brush_thickness - 2, MIN_BRUSH_THICKNESS)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("👋 Application closed successfully.")


if __name__ == "__main__":
    main()
