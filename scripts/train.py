import torch
from ultralytics import YOLO

def main():

    # 🔥 Stability settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=512,     # 🔥 reduce from 640 → 512
        batch=4,
        device=0,
        workers=2,     # 🔥 reduce workers
        amp=False      # 🔥 TURN OFF AMP
    )

if __name__ == "__main__":
    main()