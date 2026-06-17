import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

model = YOLO("best.pt")

st.title("🔒 Floutage plaques YOLO")

mode = st.sidebar.radio(
    "Mode",
    ["Image", "Vidéo", "Webcam"]
)

# =========================
# FONCTION FLOUTAGE
# =========================
def blur_plates(frame):

    results = model(
    frame,
    conf=0.15,      #plus il est petit plus il reconnais les objets difficiles ou à moitié mais a beaucoup de faux posififs
    iou=0.45,   #élimine la double détection
    imgsz=928,  #plus il est grand plus détecte les petites plaques mais rend l'algorithme très lent   
    verbose=False
)
    nb_plaques = len(results[0].boxes)
    cv2.putText(frame,
                    f"nombre de plaque{nb_plaques}",
                    (frame.shape[1] - 250, 30),  # position (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,     # police
                    0.50,                            # taille
                    (255, 20, 255),             # couleur  (BGR)
        2,                           # épaisseur
)

    for box in results[0].boxes.xyxy.cpu().numpy():

        x1, y1, x2, y2 = map(int, box[:4])
        

        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

    return frame


# =========================
# IMAGE (glisser-déposer)
# =========================
if mode == "Image":

    img_file = st.file_uploader(
        "📤 Glisser-déposer une image",
        type=["jpg", "jpeg", "png"]
    )

    if img_file:

        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


        img = blur_plates(img)


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img)


# =========================
# VIDÉO (mp4 avi mov mkv)
# =========================
elif mode == "Vidéo":

    video_file = st.file_uploader(
        "Glisser-déposer une vidéo",
        type=["mp4", "mkv", "avi", "mov"]
    )

    if video_file:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        frame_placeholder = st.empty()

        skip = st.slider("Skip frames (perf)", 1, 5, 1)

        i = 0

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            i += 1
            if i % skip != 0:
                continue

            frame = blur_plates(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(frame)

        cap.release()


# =========================
# WEBCAM LIVE
# =========================
elif mode == "Webcam":

    run = st.checkbox("▶️ Démarrer webcam")

    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()
        if not ret:
            st.error("Webcam non disponible")
            break

        frame = blur_plates(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame)

    cap.release()