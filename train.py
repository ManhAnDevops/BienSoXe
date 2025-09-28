from ultralytics import YOLO

def main():
    model = YOLO("models/yolo11n.pt")
    model.train(
        data="data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=64,       # tăng từ 16 → 32        
        device=0,   # GPU
        workers=0   # tránh multiprocessing trên Windows
    )

if __name__ == "__main__":
    main()
