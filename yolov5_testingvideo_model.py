import torch
import cv2
import math

                            #yolo5 file path                 #model file path
model = torch.hub.load('/yolov5', 'custom', path='/best.pt', source='local')

video_path = "/Video.mp4"
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

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    results.render()
    img_with_preds = results.ims[0]
    img_with_preds = cv2.cvtColor(img_with_preds, cv2.COLOR_RGB2BGR)
    cv2.imshow('Yangın Tespiti Tahmini', img_with_preds)
    out.write(img_with_preds)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()