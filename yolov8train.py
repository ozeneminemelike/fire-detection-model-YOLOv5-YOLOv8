from ultralytics import YOLO


model = YOLO("yolov8n.pt")  #which yolov8 model to use

model.train(
    data="/data.yaml",   #data.yaml file path
    epochs=25,
    imgsz=640,
    batch=8,
    name="fire-detection-model-adamW",
    pretrained=True,
    optimizer="AdamW",
    patience=10
)
