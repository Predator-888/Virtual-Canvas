# **Virtual Air Canvas: Product Requirements Document (PRD)**

**Project Name:** Virtual Air Canvas

**Target Platform:** Desktop/Webcam (Local Execution)

**Primary Technologies:** Python, OpenCV, MediaPipe

**Document Version:** 1.0

## ---

**1\. Executive Summary**

The Virtual Air Canvas is an interactive computer vision application that allows users to draw digitally on their screen by moving their index finger in front of a webcam. It eliminates the need for physical input devices (like a mouse or stylus) by leveraging real-time hand landmark tracking and dynamic line rendering. This project serves as a highly visual, interactive demonstration of advanced computer vision pipelines, real-time data tracking, and UI generation within Python, making it an ideal portfolio piece for AI/ML engineering roles.

## **2\. Objectives and Goals**

> * **Core Functionality:** Accurately track a user's index finger in 3D space using a 2D webcam feed.  
> * **Low Latency:** Render drawing strokes smoothly and without noticeable lag in real-time (aiming for 30+ FPS).  
> * **Interactive UI:** Provide an intuitive on-screen interface allowing users to change brush colors and clear the canvas using physical gestures.  
> * **Robustness:** Implement logic to handle edge cases, such as multiple hands in the frame, temporary occlusion (hand leaving the frame), and accidental drawing strokes.

## **3\. Technical Stack (The Architecture)**

The application relies entirely on lightweight, local processing to ensure privacy and low latency.

| Component | Technology / Library | Purpose&nbsp;&nbsp; |
| :---- | :---- | :---- |
| **Core Programming Language** | Python 3.x | General logic execution and library integration. |
| **Video & Image Processing** | OpenCV (cv2) | Accessing webcam hardware, image manipulation, frame flipping (mirroring), rendering lines (cv2.line), and drawing the UI (rectangles, text). |
| **Machine Learning Pipeline** | Google MediaPipe (mediapipe.solutions.hands) | Provides the pre-trained deep learning model capable of detecting 21 3D hand landmarks in real-time. |
| **Data Handling** | NumPy / Python collections.deque | NumPy for array manipulation (creating the blank canvas). Deques are crucial for storing sequential (X, Y) coordinate points efficiently, allowing the system to draw continuous lines between the previous point and the current point. |

## **4\. UI/UX Detailing & Interaction Flow**

Because the application relies on physical gestures rather than clicks, the UX must be explicitly designed for spatial interaction.

### **4.1 Layout and Visuals**

> * **The Video Feed:** The primary background is a mirrored feed from the user's webcam, allowing them to see themselves and their hand position intuitively.  
> * **The Canvas Layer:** A transparent (or black) mask overlaid on the video feed where the drawn lines are rendered.  
> * **The Header UI (Color Palette):** A static bar rendered at the top of the OpenCV window using cv2.rectangle. It contains distinct interactive zones:  
  * **Clear Button:** A prominent box (e.g., labeled "CLEAR" or colored gray) on the far left.  
  * **Color Selectors:** Four distinct colored boxes adjacent to the clear button (e.g., Blue, Green, Red, Yellow).

### **4.2 Interaction Logic (The "State Machine")**

The system determines the user's intent based on the position and state of their fingers.

> 1. **Tracking State:** The system identifies Landmark 8 (Index Finger Tip). A small colored circle is rendered on this point to provide visual feedback that tracking is active.  
> 2. **Selection State:** If the index finger tip coordinates (X, Y) enter the bounding box of a UI element in the Header:  
   * The system updates the active brush color variable.  
   * If entering the "Clear" box, all data in the coordinate deques are wiped, clearing the canvas.  
   * *Crucial UX Detail:* Drawing is temporarily paused while hovering in the header to prevent accidental marks.  
> 3. **Drawing State:** If the index finger is below the Header UI, its (X, Y) coordinates are appended to the active color's deque every frame. OpenCV draws a line connecting the points in the deque.  
> 4. **Pause State:** (Optional but highly recommended) If the user raises *both* their index and middle fingers, tracking continues, but drawing stops. This allows the user to reposition their hand without drawing an unwanted line across the screen.

## **5\. Data Structures for Sequential Drawing**

To render continuous lines, the system cannot just draw points; it must connect them. This requires storing historical coordinates.

`# Example conceptual data structure using deques`  
`bpoints = [deque(maxlen=1024)]`  
`gpoints = [deque(maxlen=1024)]`  
`rpoints = [deque(maxlen=1024)]`  
`ypoints = [deque(maxlen=1024)]`

`# Indexes to manage distinct continuous strokes (lifting the finger creates a new stroke)`  
`blue_index = 0`  
`green_index = 0`  
`red_index = 0`  
`yellow_index = 0`

When the user "lifts" their finger or changes colors, the index increments, creating a new deque so a continuous line is not drawn between two disconnected shapes.

## **6\. Implementation Phases**

> 1. **Phase 1: Basic Initialization:** Set up the webcam feed using OpenCV and ensure the frames are mirrored.  
> 2. **Phase 2: Hand Tracking Integration:** Implement MediaPipe. Extract and print the (X, Y) coordinates of Landmark 8 (index tip) to the console.  
> 3. **Phase 3: UI Rendering:** Draw the color selection rectangles and clear button at the top of the video frame.  
> 4. **Phase 4: Interaction Logic:** Write conditional statements connecting the index finger coordinates to the bounding boxes of the UI.  
> 5. **Phase 5: Drawing Logic:** Implement the deques. Append coordinates when drawing and use cv2.line to render the strokes on a blank NumPy array, then merge that array with the original video feed.