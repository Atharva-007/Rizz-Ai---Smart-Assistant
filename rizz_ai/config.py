import os

CAMERA_INDEX = int(os.environ.get("RIZZ_CAMERA_INDEX", "0"))
MODEL_PATH = os.environ.get("RIZZ_YOLO_MODEL", "yolov8n.pt")
SUMMARY_FILE = os.environ.get("RIZZ_SUMMARY_FILE", "summary.json")