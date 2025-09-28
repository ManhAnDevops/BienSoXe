import cv2
import numpy as np
import torch
from ultralytics import YOLO  

from src.char_classification.model import CNN_Model
from utils_LP import character_recog_CNN, crop_n_rotate_LP
import re

# ================= CONFIG =================
Min_char = 0.01
Max_char = 0.09
video_path = "data/video/ben.mp4"
CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight.h5'
LP_weights = 'runs/detect/train/weights/best.pt'
# Load CNN
model_char = CNN_Model(trainable=False).model
model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
print("✅ CNN model loaded:", model_char.output_shape)

# Load YOLO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_LP = YOLO(LP_weights)

# ================= VIDEO =================
cap = cv2.VideoCapture(video_path)


def format_vn_plate(raw_text: str) -> str:
    """
    Chuẩn hóa biển số VN theo pattern:
    - 2 số đầu
    - 1 chữ cái
    - 4 hoặc 5 số cuối
    Chỉ giữ lại khi đủ 7 hoặc 8 ký tự
    """
    if not raw_text:
        return ""

    # Giữ lại chỉ chữ + số
    s = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

    # Pattern VN: 2 số + 1 chữ + 4/5 số
    match = re.match(r"^(\d{2})([A-Z])(\d{4,5})$", s)
    if match:
        plate = f"{match.group(1)}{match.group(2)}{match.group(3)}"
        if len(plate) in [7, 8]:   # chỉ in nếu 7 hoặc 8 ký tự
            return plate
    
    return ""   # nếu không hợp lệ thì bỏ



# Writer để lưu kết quả
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_debug.mp4", fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, source_img = cap.read()
    if not ret:
        break

    results = model_LP(source_img, imgsz=640, conf=0.25)
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
            if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                char_x.append([x, y, cw, ch])

        if not char_x:
            continue

        char_x = sorted(char_x, key=lambda x: x[0])
        strFinalString = ""

        for j, (cx, cy, cw, ch) in enumerate(char_x):
            imgROI = rotate_thresh[cy:cy + ch, cx:cx + cw]

            # Resize chuẩn 28x28
            imgROI_resized = cv2.resize(imgROI, (28, 28), cv2.INTER_AREA)
            imgROI_resized = imgROI_resized.reshape((1, 28, 28, 1))

            # Predict
            result = model_char.predict(imgROI_resized, verbose=0)
            idx = np.argmax(result, axis=1)[0]
            text = character_recog_CNN(model_char, imgROI)
            prob = result[0][idx]

            # Debug in console
            print(f"[Frame] Char {j}: idx={idx}, predict='{text}', prob={prob:.3f}")

            if text != "Background":
                strFinalString += text
                # Vẽ ký tự nhỏ trên biển
                cv2.putText(LP_rotated, text, (cx, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Vẽ kết quả cuối trên frame
        cv2.rectangle(LP_detected_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        plate_text = format_vn_plate(strFinalString)

        # Hậu xử lý biển số
        plate_text = format_vn_plate(strFinalString)

        # Chỉ hiển thị khi hợp lệ (7 hoặc 8 ký tự)
        if plate_text and len(plate_text) in [7, 8]:
            cv2.putText(LP_detected_img, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
            print(f"✅ Biển số {i}: {plate_text}")
        else:
            print(f"❌ Bỏ qua: {strFinalString}")


    # Show & Save
    cv2.imshow("Video Debug", cv2.resize(LP_detected_img, (960, 720)))
    out.write(LP_detected_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
