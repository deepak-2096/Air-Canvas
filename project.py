import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

# Initialize page config
st.set_page_config(page_title="Air Canvas", page_icon="🖌️", layout="wide")

st.title("🖌️ Air Canvas - Hand Gesture Writing App")
st.markdown("Enable your webcam and draw on the screen using hand gestures!")

# --- SIDEBAR SETTINGS ---
st.sidebar.title("App Settings")

st.sidebar.markdown("### 🎨 Pen Properties")
colors_map = {
    "Red": (0, 0, 255),    # BGR format for OpenCV
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Cyan": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "White": (255, 255, 255)
}

selected_color = st.sidebar.selectbox("Select Color", list(colors_map.keys()))
brush_size = st.sidebar.slider("Brush Size", 2, 30, 5)
eraser_size = st.sidebar.slider("Eraser Size", 20, 150, 70)

st.sidebar.markdown("---")
clear_btn = st.sidebar.button("🗑️ Clear Canvas")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖐️ Gesture Guide")
st.sidebar.markdown("""
- **Draw**: *Only Index Finger Up*
- **Hover/Move**: *Index & Middle Fingers Up* (Moves cursor without drawing)
- **Erase**: *All 4 Fingers Up* (Acts as a large eraser)
""")


# --- WEBRTC VIDEO PROCESSOR ---
class HandTrackingProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize MediaPipe inside the thread
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.75, 
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # State variables
        self.canvas = None
        self.x_prev = 0
        self.y_prev = 0
        
        # UI controls (updated from main thread)
        self.color = colors_map["Red"]
        self.brush_size = 5
        self.eraser_size = 70
        self.clear_canvas = False

    def recv(self, frame_container):
        # frame_container is an av.VideoFrame
        frame = frame_container.to_ndarray(format="bgr24")
        
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Initialize the canvas if not done yet
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            
        # Handle manual clear request from the UI
        if self.clear_canvas:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self.clear_canvas = False # reset flag
            
        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get the pixel coordinates for specific landmarks
                landmarks = hand_landmarks.landmark
                
                # We need x, y coordinates
                lm_list = []
                for id, lm in enumerate(landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    
                if len(lm_list) != 0:
                    # Index and Middle finger tips
                    x1, y1 = lm_list[8][1], lm_list[8][2]
                    x2, y2 = lm_list[12][1], lm_list[12][2]
                    
                    # Check which fingers are up
                    fingers = []
                    # Tip IDs for Index, Middle, Ring, Pinky
                    tip_ids = [8, 12, 16, 20]
                    # PIP IDs for Index, Middle, Ring, Pinky
                    pip_ids = [6, 10, 14, 18]
                    
                    for tip, pip in zip(tip_ids, pip_ids):
                        if lm_list[tip][2] < lm_list[pip][2]:  # y is lower = finger is up
                            fingers.append(1)
                        else:
                            fingers.append(0)
                            
                    # 1. Hover Mode: Index and Middle fingers are UP
                    if len(fingers) >= 2 and fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0:
                        self.x_prev, self.y_prev = x1, y1
                        
                        # Draw pointer indicator
                        cv2.circle(frame, (x1, y1), 15, self.color, cv2.FILLED)
                        
                    # 2. Draw Mode: Only Index finger is UP
                    elif len(fingers) >= 2 and fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
                        cv2.circle(frame, (x1, y1), self.brush_size, self.color, cv2.FILLED)
                        
                        if self.x_prev == 0 and self.y_prev == 0:
                            self.x_prev, self.y_prev = x1, y1
                            
                        # Draw on the canvas
                        cv2.line(self.canvas, (self.x_prev, self.y_prev), (x1, y1), self.color, self.brush_size)
                        
                        self.x_prev, self.y_prev = x1, y1
                        
                    # 3. Erase Mode: All 4 fingers UP
                    elif sum(fingers) == 4:
                        # Find center of palm roughly (using landmark 9)
                        cx, cy = lm_list[9][1], lm_list[9][2]
                        cv2.circle(frame, (cx, cy), self.eraser_size, (50, 50, 50), cv2.FILLED)
                        # Erase by drawing black on canvas
                        cv2.circle(self.canvas, (cx, cy), self.eraser_size, (0, 0, 0), cv2.FILLED)
                        
                    else:
                        # Any other gesture -> reset previous points to break continuous drawing line
                        self.x_prev, self.y_prev = 0, 0

        # Combine the original frame and the canvas
        # Method: convert canvas to grayscale, create inverse mask, apply mask to frame, add canvas
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        
        # Black out the regions where drawing exists on the video frame
        frame = cv2.bitwise_and(frame, img_inv)
        
        # Add the colored canvas drawing onto the video frame
        frame = cv2.bitwise_or(frame, self.canvas)
        
        return av.VideoFrame.from_ndarray(frame, format="bgr24")


# WebRTC requires STUN servers to work reliably when deployed to the cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- RUN WEBRTC STREAM ---
webrtc_ctx = webrtc_streamer(
    key="air-canvas",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=HandTrackingProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Update the processor instance based on UI inputs
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.color = colors_map[selected_color]
    webrtc_ctx.video_processor.brush_size = brush_size
    webrtc_ctx.video_processor.eraser_size = eraser_size
    
    if clear_btn:
        webrtc_ctx.video_processor.clear_canvas = True

st.markdown("---")
st.markdown("### How to Use:")
st.markdown("1. Click **START** to open your webcam.")
st.markdown("2. Grant the browser permission to access your camera.")
st.markdown("3. Show your hand to the camera and use gestures (listed on the sidebar) to draw!")

