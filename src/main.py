from ultralytics import YOLO

MODEL_PATH = 'E:\AIO\Helmet_Safety_Detection\Data_test\test.pt'
model = YOLO( MODEL_PATH )

YAML_PATH ='E:\AIO\Helmet_Safety_Detection\Safety_Helmet_Dataset\data.yaml'
EPOCHS = 20
IMG_SIZE = 64
BATCH_SIZE = 32

model.info()

model.train(data = YAML_PATH ,
            epochs =EPOCHS ,
            batch = BATCH_SIZE ,
            imgsz = IMG_SIZE )