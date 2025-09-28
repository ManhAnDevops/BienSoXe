from ultralytics import YOLO
import torch

device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)

print("Device", device)
