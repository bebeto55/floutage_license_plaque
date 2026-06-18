import streamlit as st
import cv2
import numpy as np
import tempfile
import av


from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ====================================================
# MODEL
# ====================================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🔒 Floutage plaques YOLO")

mode = st.sidebar.radio(
    "Mode",
    ["Image", "Vidéo", "Webcam", "Téléphone"]
)

# ====================================================
# FLOUTAGE
# ====================================================

def blur_plates(frame):

    results = model(
        frame,
        conf=0.15,
        iou=0.45,
        imgsz=928,
        verbose=False
    )

    nb_plaques = len(results[0].boxes)

    cv2.putText(
        frame,
        f"Plaques : {nb_plaques}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 20, 255),
        2
    )

    for box in results[0].boxes.xyxy.cpu().numpy():

        x1, y1, x2, y2 = map(int, box[:4])

        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(
                roi,
                (51, 51),
                30
            )

    return frame

# ====================================================
# IMAGE
# ====================================================

if mode == "Image":

    img_file = st.file_uploader(
        "📤 Glisser-déposer une image",
        type=["jpg", "jpeg", "png"]
    )

    if img_file:

        file_bytes = np.asarray(
            bytearray(img_file.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        img = blur_plates(img)

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )

# ====================================================
# VIDEO
# ====================================================

elif mode == "Vidéo":

    video_file = st.file_uploader(
        "📤 Glisser-déposer une vidéo",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if video_file:

        skip = st.slider(
            "Skip frames",
            1,
            5,
            1
        )

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()

        i = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            i += 1

            if i % skip != 0:
                continue

            frame = blur_plates(frame)

            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

        cap.release()

# ====================================================
# WEBCAM / TELEPHONE
# ====================================================

elif mode in ["Webcam", "Téléphone"]:

    skip = st.slider(
        "Skip frames (perf)",
        1,
        5,
        3
    )

    class PlateProcessor(VideoProcessorBase):

        def __init__(self):
            self.frame_count = 0
            self.last_frame = None

        def recv(self, frame):

            img = frame.to_ndarray(format="bgr24")

            self.frame_count += 1

            if self.frame_count % skip == 0:

                self.last_frame = blur_plates(img)

            if self.last_frame is None:
                self.last_frame = img

            return av.VideoFrame.from_ndarray(
                self.last_frame,
                format="bgr24"
            )

    constraints = {
        "video": True,
        "audio": False,
    }

    if mode == "Téléphone":
        constraints = {
            "video": {
                "facingMode": "environment"
            },
            "audio": False,
        }

    webrtc_streamer(
        key=f"camera-{mode}",
        video_processor_factory=PlateProcessor,
        media_stream_constraints=constraints,
        async_processing=True,
    )
