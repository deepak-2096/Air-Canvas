import threading
import os
import urllib.request
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Air Canvas", page_icon="🖌️", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
  .stApp                      { background: #0f0f1a; color: #e0e0ff; }
  .stSidebar                  { background: #1a1a2e; }
  h1 {
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem; font-weight: 700;
  }
</style>
""", unsafe_allow_html=True)

st.title("🖌️ Air Canvas — Hand Gesture Drawing")
st.markdown("Draw in mid-air using hand gestures!")

# ─── DOWNLOAD MODEL IF NEEDED ─────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

@st.cache_resource(show_spinner="Downloading hand tracking model…")
def download_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

model_path = download_model()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("### 🎨 Pen Color")
    colors_map = {
        "🔴 Red":    (0,   0,   255),
        "🟢 Green":  (0,   255, 0),
        "🔵 Blue":   (255, 0,   0),
        "🟡 Yellow": (0,   255, 255),
        "🟣 Purple": (180, 0,   180),
        "⚪ White":  (255, 255, 255),
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
**✋ Hover** — Index + Middle (peace ✌️)  
**🧹 Erase** — All 4 fingers open  
    """)

# ─── RTC CONFIG ───────────────────────────────────────────────────────────────
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Landmark index constants
WRIST        = 0
INDEX_TIP    = 8;  INDEX_PIP    = 6
MIDDLE_TIP   = 12; MIDDLE_PIP   = 10
RING_TIP     = 16; RING_PIP     = 14
PINKY_TIP    = 20; PINKY_PIP    = 18
PALM_CENTER  = 9

# Finger connections for drawing skeleton manually
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (5,9),(9,10),(10,11),(11,12),     # middle
    (9,13),(13,14),(14,15),(15,16),   # ring
    (13,17),(17,18),(18,19),(19,20),  # pinky
    (0,17),                           # palm
]

# ─── VIDEO PROCESSOR ──────────────────────────────────────────────────────────
class HandTrackingProcessor(VideoProcessorBase):
    def __init__(self):
        self._lock           = threading.Lock()
        self._mp_initialized = False

        self.canvas   = None
        self.x_prev   = 0
        self.y_prev   = 0

        # Settings written by main thread
        self.color        = (0, 0, 255)
        self.brush_size   = 6
        self.eraser_size  = 70
        self.clear_canvas = False

    def _ensure_mp(self):
        """Initialize MediaPipe HandLandmarker inside the worker thread."""
        if self._mp_initialized:
            return
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._mp_initialized = True

    def _fingers_up(self, pts):
        """Return [index, middle, ring, pinky] = 1 if finger is extended."""
        pairs = [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                 (RING_TIP, RING_PIP),   (PINKY_TIP,  PINKY_PIP)]
        return [1 if pts[tip][1] < pts[pip][1] else 0 for tip, pip in pairs]

    def _draw_skeleton(self, frame, pts, color=(0, 255, 0)):
        """Draw hand skeleton lines and landmark dots on frame."""
        for (a, b) in HAND_CONNECTIONS:
            cv2.line(frame, (pts[a][0], pts[a][1]),
                     (pts[b][0], pts[b][1]), color, 2)
        for i, (x, y) in enumerate(pts):
            dot_color = (255, 255, 255) if i in [4,8,12,16,20] else (100, 200, 255)
            cv2.circle(frame, (x, y), 5, dot_color, cv2.FILLED)

    def recv(self, frame_av):
        # 1. Decode & flip frame
        frame = frame_av.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # 2. Lazy init inside worker thread
        self._ensure_mp()

        # 3. Canvas init / resize
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # 4. Thread-safe settings read
        with self._lock:
            color       = self.color
            brush_size  = self.brush_size
            eraser_size = self.eraser_size
            do_clear    = self.clear_canvas
            if do_clear:
                self.clear_canvas = False

        if do_clear:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self.x_prev = self.y_prev = 0

        # 5. Run hand detection (Tasks API uses mp.Image)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = self._detector.detect(mp_image)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                # Convert normalized to pixel coords
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

                # Draw skeleton
                self._draw_skeleton(frame, pts)

                # Finger state (use y-coord: smaller y = higher on screen = UP)
                def up(tip, pip): return pts[tip][1] < pts[pip][1]
                idx_up    = up(INDEX_TIP,  INDEX_PIP)
                mid_up    = up(MIDDLE_TIP, MIDDLE_PIP)
                ring_up   = up(RING_TIP,   RING_PIP)
                pinky_up  = up(PINKY_TIP,  PINKY_PIP)

                ix, iy = pts[INDEX_TIP]

                # ── HOVER: index + middle up ─────────────────────────────
                if idx_up and mid_up and not ring_up:
                    cv2.circle(frame, (ix, iy), 14, color, cv2.FILLED)
                    cv2.circle(frame, (ix, iy), 16, (255, 255, 255), 2)
                    self.x_prev, self.y_prev = ix, iy

                # ── DRAW: only index up ──────────────────────────────────
                elif idx_up and not mid_up:
                    cv2.circle(frame, (ix, iy), brush_size + 2, color, cv2.FILLED)
                    if self.x_prev == 0 and self.y_prev == 0:
                        self.x_prev, self.y_prev = ix, iy
                    cv2.line(self.canvas,
                             (self.x_prev, self.y_prev), (ix, iy),
                             color, brush_size)
                    self.x_prev, self.y_prev = ix, iy

                # ── ERASE: all 4 fingers up ──────────────────────────────
                elif idx_up and mid_up and ring_up and pinky_up:
                    cx, cy = pts[PALM_CENTER]
                    cv2.circle(frame,       (cx, cy), eraser_size, (60, 60, 60),    cv2.FILLED)
                    cv2.circle(frame,       (cx, cy), eraser_size, (220, 220, 220), 2)
                    cv2.circle(self.canvas, (cx, cy), eraser_size, (0, 0, 0),       cv2.FILLED)
                    self.x_prev = self.y_prev = 0

                else:
                    self.x_prev = self.y_prev = 0

        # 6. Blend canvas onto frame
        gray        = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
        inv_mask    = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
        frame       = cv2.bitwise_and(frame, inv_mask)
        frame       = cv2.bitwise_or(frame, self.canvas)

        return av.VideoFrame.from_ndarray(frame, format="bgr24")


# ─── STREAM ───────────────────────────────────────────────────────────────────
webrtc_ctx = webrtc_streamer(
    key="air-canvas",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=HandTrackingProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)

# Push sidebar values into the processor safely
if webrtc_ctx.video_processor:
    with webrtc_ctx.video_processor._lock:
        webrtc_ctx.video_processor.color       = colors_map[selected_color]
        webrtc_ctx.video_processor.brush_size  = brush_size
        webrtc_ctx.video_processor.eraser_size = eraser_size
        if clear_btn:
            webrtc_ctx.video_processor.clear_canvas = True

# ─── HOW TO USE ───────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### ✏️ Draw")
    st.markdown("Raise only your **index finger** and move to draw on the canvas.")
with c2:
    st.markdown("### ✋ Hover")
    st.markdown("Raise **index + middle** (peace ✌️) to move without drawing.")
with c3:
    st.markdown("### 🧹 Erase")
    st.markdown("Open all **4 fingers** wide to erase the area under your palm.")
