import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Air Canvas", page_icon="🖌️", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #0f0f1a; }
  .stApp { background: #0f0f1a; color: #e0e0ff; }
  .stSidebar { background: #1a1a2e; }
  h1 { 
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem; font-weight: 700;
  }
</style>
""", unsafe_allow_html=True)

st.title("🖌️ Air Canvas — Hand Gesture Drawing")
st.markdown("Draw in mid-air using hand gestures! Point your index finger to draw, show peace sign to hover, open all fingers to erase.")

# ─── SIDEBAR SETTINGS ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("### 🎨 Pen Color")
    colors_map = {
        "🔴 Red":     (0,   0,   255),
        "🟢 Green":   (0,   255, 0),
        "🔵 Blue":    (255, 0,   0),
        "🟡 Yellow":  (0,   255, 255),
        "🟣 Purple":  (180, 0,   180),
        "⚪ White":   (255, 255, 255),
    }
    selected_color = st.selectbox("Color", list(colors_map.keys()), index=0)
    brush_size     = st.slider("Brush Size",  2,  40,  6)
    eraser_size    = st.slider("Eraser Size", 20, 150, 70)

    st.markdown("---")
    clear_btn = st.button("🗑️ Clear Canvas", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🖐️ Gesture Guide")
    st.info("""
**✏️ Draw** — Only index finger up  
**✋ Hover** — Index + Middle fingers up  
**🧹 Erase** — All 4 fingers up  
    """)

# ─── RTC CONFIG ──────────────────────────────────────────────────────────────────
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ─── VIDEO PROCESSOR ─────────────────────────────────────────────────────────────
class HandTrackingProcessor(VideoProcessorBase):
    def __init__(self):
        # ── DO NOT init MediaPipe here — it must run inside the worker thread ──
        self._mp_initialized = False

        # Canvas state
        self.canvas  = None
        self.x_prev  = 0
        self.y_prev  = 0

        # Shared settings (written by main thread, read by worker thread)
        self.color        = (0, 0, 255)
        self.brush_size   = 6
        self.eraser_size  = 70
        self.clear_canvas = False

    def _init_mediapipe(self):
        """Lazy init — called once inside the worker thread on first frame."""
        self._mp_hands = mp.solutions.hands
        self._hands    = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )
        self._mp_draw     = mp.solutions.drawing_utils
        self._mp_initialized = True

    def _fingers_up(self, lm_list):
        """Return list [index, middle, ring, pinky] = 1 if finger extended."""
        tip_ids = [8,  12, 16, 20]
        pip_ids = [6,  10, 14, 18]
        fingers = []
        for tip, pip in zip(tip_ids, pip_ids):
            fingers.append(1 if lm_list[tip][2] < lm_list[pip][2] else 0)
        return fingers

    def recv(self, frame_av):
        # ── Lazy MediaPipe init inside worker thread ──
        if not self._mp_initialized:
            self._init_mediapipe()

        frame = frame_av.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Init canvas on first frame (or after size change)
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Handle clear request from sidebar button
        if self.clear_canvas:
            self.canvas       = np.zeros((h, w, 3), dtype=np.uint8)
            self.clear_canvas = False

        # ── MediaPipe hand detection ──
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(frame, hand_lms, self._mp_hands.HAND_CONNECTIONS)

                # Build landmark pixel list
                lm_list = []
                for idx, lm in enumerate(hand_lms.landmark):
                    lm_list.append([idx, int(lm.x * w), int(lm.y * h)])

                if len(lm_list) < 21:
                    continue

                fingers = self._fingers_up(lm_list)
                x1, y1  = lm_list[8][1],  lm_list[8][2]   # Index fingertip

                # ── HOVER mode: index + middle up ──────────────────────────
                if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0:
                    cv2.circle(frame, (x1, y1), 12, self.color, cv2.FILLED)
                    cv2.circle(frame, (x1, y1), 14, (255,255,255), 2)
                    self.x_prev, self.y_prev = x1, y1  # update anchor without drawing

                # ── DRAW mode: only index up ───────────────────────────────
                elif fingers[0] == 1 and fingers[1] == 0:
                    cv2.circle(frame, (x1, y1), self.brush_size, self.color, cv2.FILLED)
                    if self.x_prev == 0 and self.y_prev == 0:
                        self.x_prev, self.y_prev = x1, y1
                    cv2.line(self.canvas,
                             (self.x_prev, self.y_prev),
                             (x1, y1),
                             self.color, self.brush_size)
                    self.x_prev, self.y_prev = x1, y1

                # ── ERASE mode: all 4 fingers up ──────────────────────────
                elif sum(fingers) == 4:
                    cx, cy = lm_list[9][1], lm_list[9][2]  # palm centre
                    cv2.circle(frame,        (cx, cy), self.eraser_size, (40, 40, 40), cv2.FILLED)
                    cv2.circle(frame,        (cx, cy), self.eraser_size, (200,200,200), 2)
                    cv2.circle(self.canvas,  (cx, cy), self.eraser_size, (0, 0, 0), cv2.FILLED)
                    self.x_prev, self.y_prev = 0, 0

                # ── Other gestures: break drawing line ────────────────────
                else:
                    self.x_prev, self.y_prev = 0, 0

        # ── Blend canvas onto frame ──────────────────────────────────────────
        gray        = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
        inv_mask    = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
        frame       = cv2.bitwise_and(frame, inv_mask)
        frame       = cv2.bitwise_or(frame, self.canvas)

        return av.VideoFrame.from_ndarray(frame, format="bgr24")


# ─── WEBRTC STREAM ────────────────────────────────────────────────────────────────
webrtc_ctx = webrtc_streamer(
    key="air-canvas",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=HandTrackingProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Push sidebar settings into the processor safely
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.color       = colors_map[selected_color]
    webrtc_ctx.video_processor.brush_size  = brush_size
    webrtc_ctx.video_processor.eraser_size = eraser_size
    if clear_btn:
        webrtc_ctx.video_processor.clear_canvas = True

# ─── HOW TO USE ──────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### ✏️ Draw")
    st.markdown("Raise only your **index finger** and move your hand to draw on the canvas.")
with col2:
    st.markdown("### ✋ Hover")
    st.markdown("Raise **index + middle fingers** (peace sign) to move without drawing.")
with col3:
    st.markdown("### 🧹 Erase")
    st.markdown("Open all **4 fingers** wide to erase the area under your palm.")
