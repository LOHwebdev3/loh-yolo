import cv2
import math
from ultralytics import solutions
import logging

from ObjectCounter import ObjectCounter
from TheCounter import TheCounter

# 📂 ตั้งค่าไฟล์วิดีโอและโมเดล
video_path = "C:/Users/loh-ai/Documents/loh-yolo/vdo/12hr.mp4"
model_path = "runs/detect/train14/weights/best.pt"

# 🛑 Suppress warnings
logging.getLogger('ultralytics').setLevel(logging.ERROR)

if __name__ == '__main__':
    TheCounter(name='test', video_path=video_path, model_path=model_path,_show=True)