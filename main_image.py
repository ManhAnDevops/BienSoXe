import cv2
import numpy as np
import torch
import os

from src.char_classification.model import CNN_Model
from utils_LP import character_recog_CNN, crop_n_rotate_LP
from ultralytics import YOLO  

# ==== CONFIG ====
Min_char = 0.01
Max_char = 0.09
image_path = 'data/test/images/clip3.2_new_0.jpg'
CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight.h5'
LP_weights = 'runs/detect/train/weights/best.pt'

# Thư mục debug
os.makedirs("debug_chars", exist_ok=True)

# ==== LOAD MODELS ====
model_char = CNN_Model(trainable=False).model
model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_LP = YOLO(LP_weights)

# ==== RUN YOLO DETECT ====
source_img = cv2.imread(image_path)
results = model_LP(source_img, imgsz=640, conf=0.25)
LP_detected_img = results[0].plot(labels=False)  
print(f"YOLO detect {len(results[0].boxes)} box(es).")

c = 0
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    conf = float(box.conf[0])
    print(f"👉 Box {c}: {x1},{y1},{x2},{y2}, conf={conf:.2f}")

    angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(source_img, x1, y1, x2, y2)
    if (rotate_thresh is None) or (LP_rotated is None):
        print("❌ Không crop được biển số.")
        continue

    cv2.imshow(f'LP_rotated_{c}', LP_rotated)
    cv2.imshow(f'rotate_thresh_{c}', rotate_thresh)

    # ==== Contour ====
    LP_rotated_copy = LP_rotated.copy()
    cont, _ = cv2.findContours(rotate_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(f"⚡ Tổng contour tìm thấy: {len(cont)}")
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:17]

    char_x = []
    height, width, _ = LP_rotated_copy.shape
    roiarea = height * width

    for ind, cnt in enumerate(cont):
        (x, y, w, h) = cv2.boundingRect(cnt)
        ratiochar = w / h
        char_area = w * h
        print(f"  Contour {ind}: x={x},y={y},w={w},h={h}, area={char_area}, ratio={ratiochar:.2f}")

        # Lưu ảnh để check
        roi = rotate_thresh[y:y+h, x:x+w]
        cv2.imwrite(f"debug_chars/char_{c}_{ind}.png", roi)

        if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
            char_x.append([x, y, w, h])

    if not char_x:
        print("❌ Không tìm thấy ký tự hợp lệ trong box.")
        continue

    char_x = sorted(char_x, key=lambda x: x[0])
    strFinalString = ""
    first_line = ""
    second_line = ""
    threshold_12line = np.min([cx[1] for cx in char_x]) + (np.mean([cx[3] for cx in char_x]) / 2)

    for i, (x, y, w, h) in enumerate(char_x):
        roi = rotate_thresh[y:y+h, x:x+w]
        text = character_recog_CNN(model_char, roi)
        print(f"   ➡ Char {i}: predict={text}")

        if text == 'Background':
            text = ''

        if y < threshold_12line:
            first_line += text
        else:
            second_line += text

    strFinalString = first_line + second_line
    print(f"✅ Biển số đọc được: {strFinalString}")
    cv2.putText(LP_detected_img, strFinalString, (x1, y1 - 20), 
                  cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)

    c += 1

cv2.imshow('input', cv2.resize(source_img, dsize=None, fx=0.5, fy=0.5))
cv2.imshow('final_result', cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5))
print("Finally Done!")
cv2.waitKey(0)
