import cv2
import numpy as np
from src.char_classification.model import CNN_Model

model = CNN_Model(trainable=False).model
model.load_weights("src/weights/weight.h5")

# tạo input giả (ảnh trắng 28x28)
img = np.ones((28, 28, 1), dtype=np.float32)
img = np.expand_dims(img, axis=0)  # (1,28,28,1)

pred = model.predict(img)
print("Pred shape:", pred.shape)
print("Pred:", pred)
print("Class:", np.argmax(pred))
