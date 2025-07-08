import os
from yolov5 import train

cache_path = 'path/to/your/cache'
if os.path.exists(cache_path):
    os.remove(cache_path)

if __name__ == '__main__':
    train.run(
        data="/data.yaml",
        epochs=25,
        imgsz=320,
        batch_size=8,
        name="fire-detection-model-yolov5_320_sgd",
        pretrained=True,
        patience=10
    )