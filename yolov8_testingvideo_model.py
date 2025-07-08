from ultralytics import YOLO
import cv2
import math

model = YOLO("/best.pt") # model file path

video_path = "/Video.mp4"  # video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video açılamadı.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

delay = math.ceil(1000 / fps)

output_path = "/video_predict.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(source=frame, verbose=False)
    img_with_preds = results[0].plot()
    cv2.imshow('Yangın Tespiti Tahmini', img_with_preds)
    out.write(img_with_preds)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
