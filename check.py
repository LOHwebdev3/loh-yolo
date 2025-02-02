from playsound import playsound
from ultralytics import YOLO
import torch
try:
    model = YOLO("yolo11n.pt")

    print("torch " + torch.__version__)  # PyTorch version
    if torch.cuda.is_available():
        print("CUDA IS SUPPORT")  # Should return True if CUDA is available
        print(torch.cuda.get_device_name(0))  # Name of the GPU
        print(torch.cuda.get_device_name(1))  # Name of the GPU
    else:
        print("CUDA IN NOT SUPPORTED")

    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    elif torch.backends.mps.is_available():
        print("MPS is available: Using Metal Performance Shaders (MPS) on macOS.")
    else:
        print("No GPU available. Falling back to CPU.")
except Exception as e:
    print(e)

