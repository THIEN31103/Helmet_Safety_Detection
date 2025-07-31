import cv2
from ultralytics import YOLO

TRAINED_MODEL_PATH =r'E:\AIO\Helmet_Safety_Detection\src\runs\detect\train\weights\best.pt'
model = YOLO(TRAINED_MODEL_PATH)

YAML_PATH ='E:\AIO\Helmet_Safety_Detection\Safety_Helmet_Dataset\data.yaml'
IMG_SIZE = 64
model.val(data=YAML_PATH,
          imgsz=IMG_SIZE,
          split='test')
