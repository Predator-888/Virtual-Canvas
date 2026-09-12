# 🎨 Virtual Air Canvas

An interactive, real-time Computer Vision application that allows users to paint and manipulate digital content in mid-air using webcam hand tracking and spatial gesture recognition.

---

## 🌟 Key Features

- **Intuitive Touchless Drawing**: Draw smoothly with your index finger in mid-air with sub-pixel anti-aliasing and trajectory smoothing.
- **Dynamic 4-Finger Movement Eraser**: Simply raise 4 fingers and move your hand to cleanly erase drawings along your hand's path.
- **Whole-Palm Canvas Panning**: Open your palm (all 5 fingers) to translate and slide your entire artwork across the screen.
- **Gesture-Driven UI Toolbar**: Hover over color swatches and the `CLEAR` button using a 2-finger navigation gesture.
- **Safety Header Guard**: Prevents drawing strokes from spilling over the top navigation toolbar.
- **Dual View Modes**: Seamlessly toggle between AR camera overlay and a distraction-free digital canvas board.
- **Snapshot Export**: Save your artwork directly to timestamped high-resolution PNG images.

---

## 🖐️ Spatial Gesture Controls

| Gesture | Posture | Action |
| :--- | :--- | :--- |
| **Draw** | 👉 **1 Finger** (Index up) | Smooth continuous drawing in active color |
| **Select / Hover** | ✌️ **2 Fingers** (Index + Middle) | Hover over UI elements, select colors, click `CLEAR` without drawing |
| **Erase** | 🖐️ **4 Fingers** (Index, Middle, Ring, Pinky) | Motion-based eraser follows hand movement and erases content |
| **Move Canvas** | ✋ **5 Fingers** (Open Palm) | Pan and drag the entire drawing canvas across the screen |
| **Idle** | ✊ **Fist / Inactive** | Stops drawing; lifts virtual pen |

---

## ⌨️ Keyboard Shortcuts

- `C` : Clear Canvas
- `S` : Save Snapshot to `snapshots/` folder
- `V` : Toggle View (AR Camera Feed vs. Pure Canvas Board)
- `+` / `=` : Increase brush thickness
- `-` / `_` : Decrease brush thickness
- `Q` / `ESC` : Quit Application

---

## 🏗️ Architecture & Modules

```
Virtual-Canvas/
├── src/
│   ├── config.py           # Resolution, color palettes, UI button boundaries, defaults
│   ├── hand_tracker.py     # MediaPipe Hands wrapper, 21-landmark tracking & gesture classifier
│   ├── ui.py               # Glassmorphic header toolbar, badges, FPS counter, context cursors
│   └── canvas.py           # Canvas buffer, stroke smoothing, continuous eraser, affine pan, blending
├── main.py                 # Main capture loop, gesture dispatcher, event loop
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python main.py
```