from ultralytics import YOLO

# ใช้ if __name__ == '__main__' เพื่อแก้ปัญหานี้
if __name__ == '__main__':
    # โหลดโมเดล YOLOv8
    model = YOLO("yolo11n.pt").to('cuda')  # คุณสามารถเลือกขนาดโมเดลได้ เช่น yolov8n.pt, yolov8s.pt

    # Train the model with GPUs
    results = model.train(data="datasets/data.yaml", epochs=100, imgsz=640, device=[0])
