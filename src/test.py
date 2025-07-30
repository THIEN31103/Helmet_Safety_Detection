import cv2
from ultralytics import YOLO

TRAINED_MODEL_PATH ='E:\AIO\Helmet_Safety_Detection\src\runs\detect\train\weights\best.pt'
model = YOLO(TRAINED_MODEL_PATH)

IMAGE_PATH ='E:\AIO\Helmet_Safety_Detection\Data_test\img.png'
CONF_THRESHOLD = 0.3
IMAGE_SIZE = 64

results = model.predict(source= IMAGE_PATH,
                        imgsz=IMAGE_SIZE,
                        conf= CONF_THRESHOLD)
img_results = results[0].plot()

cv2.imshow(img_results)