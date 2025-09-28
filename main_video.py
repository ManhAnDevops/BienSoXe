import cv2
import numpy as np
import torch
from ultralytics import YOLO  
import re

from src.char_classification.model import CNN_Model
from utils_LP import character_recog_CNN, crop_n_rotate_LP
from collections import deque

# ================= CONFIG =================
video_path = "data/video/ben15.mp4"
CHAR_CLASSIFICATION_WEIGHTS = "./src/weights/weight.h5"
LP_weights = "runs/detect/train/weights/best.pt"

stable_plate = None
stable_count = 0
buffer = deque(maxlen=5)

# Load CNN
model_char = CNN_Model(trainable=False).model
model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
print("✅ CNN model loaded:", model_char.output_shape)

# Load YOLO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_LP = YOLO(LP_weights)



# Bộ nhớ đệm 5 lần gần nhất
plate_buffer = deque(maxlen=5)
stable_plate = ""   # plate đã được xác nhận ổn định

def update_plate(new_plate: str) -> str:
    """
    - Luôn trả về plate để vẽ (dù sai).
    - Nếu một plate xuất hiện >=5 lần liên tiếp => stable_plate.
    - Sau đó nếu detect sai => vẫn vẽ stable_plate.
    """
    global stable_plate

    if not new_plate:
        return stable_plate  # nếu không detect ra gì thì giữ plate ổn định

    # Thêm vào buffer
    plate_buffer.append(new_plate)

    # Kiểm tra xem buffer có toàn 1 plate không
    if len(plate_buffer) == plate_buffer.maxlen and len(set(plate_buffer)) == 1:
        stable_plate = new_plate
        print(f"✅ Xác nhận plate ổn định: {stable_plate}")

    # Nếu chưa ổn định thì vẫn trả về plate detect được
    return new_plate if new_plate else stable_plate


# ================= Hàm chuẩn hóa biển số =================
def format_vn_plate(raw_text: str) -> str:
    if not raw_text:
        return ""

    s = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if len(s) < 3:
        return ""

    chars = list(s)

    for i in [0, 1]:
        if i < len(chars) and not chars[i].isdigit():
            chars[i] = "0"

    s = "".join(chars)

    s = re.sub(r"^(\d{2})0([A-Z])", r"\1\2", s)

    match = re.match(r"^(\d{2})([A-Z])(\d{4,5})$", s)
    if match:
        plate = f"{match.group(1)}{match.group(2)}{match.group(3)}"
        if len(plate) in [7, 8]:
            return plate
    return ""



# ================= Bộ nhớ để ổn định kết quả =================
plate_memory = {}
STABLE_THRESHOLD = 3   # cần nhận ≥5 lần mới chấp nhận khóa

# ================= VIDEO =================
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_debug.mp4", fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, source_img = cap.read()
    if not ret:
        break

    results = model_LP(source_img, conf=0.75)
    
    LP_detected_img = source_img.copy()

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Crop + rotate
        angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(source_img, x1, y1, x2, y2)
        if (rotate_thresh is None) or (LP_rotated is None):
            continue

        # Contours tách ký tự
        cont, _ = cv2.findContours(rotate_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cont = sorted(cont, key=cv2.contourArea, reverse=True)[:17]

        char_x = []
        h, w, _ = LP_rotated.shape
        roiarea = h * w
        for cnt in cont:
            x, y, cw, ch = cv2.boundingRect(cnt)
            ratiochar = cw / ch
            char_area = cw * ch
            if (0.005 * roiarea < char_area < 0.15 * roiarea) and (0.25 < ratiochar < 1.0):
                char_x.append([x, y, cw, ch])

        if not char_x:
            continue

        char_x = sorted(char_x, key=lambda x: x[0])
        strFinalString = ""

        for j, (cx, cy, cw, ch) in enumerate(char_x):
            imgROI = rotate_thresh[cy:cy + ch, cx:cx + cw]
            text = character_recog_CNN(model_char, imgROI)
            if text != "Background":
                strFinalString += text

        # Chuẩn hóa
        plate_text = format_vn_plate(strFinalString)

        if plate_text:
            buffer.append(plate_text)

            # Nếu buffer có 5 phần tử giống nhau → lock
            if len(buffer) == 5 and len(set(buffer)) == 1:
                stable_plate = plate_text
                stable_count += 1

        # Luôn vẽ
        cv2.rectangle(LP_detected_img, (x1, y1), (x2, y2), (0,255,0), 3)

        if stable_plate:
            cv2.putText(LP_detected_img, stable_plate, (x1, y1-10),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
            print(f"✅ Biển số (ổn định): {stable_plate}")
        else:
            cv2.putText(LP_detected_img, strFinalString, (x1, y1-10),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
            print(f"❌ Bỏ qua: {strFinalString}")
    # Show & Save
    cv2.imshow("Video Debug", cv2.resize(LP_detected_img, (960, 720)))
    out.write(LP_detected_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
