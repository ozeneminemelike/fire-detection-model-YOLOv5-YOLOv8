from ultralytics import YOLO
import cv2
import os

model = YOLO("/best.pt")  # Model file path

input_folder = "/test_images"  # test pics file path
output_folder = "/test_outputs"  # output pics file path

os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"{image_name} okunamadı.")
        continue


    results = model.predict(source=image, verbose=False)
    img_with_preds = results[0].plot()

    cv2.imshow("Tahmin", img_with_preds)
    cv2.waitKey(500)
    output_path = os.path.join(output_folder, f"pred_{image_name}")
    cv2.imwrite(output_path, img_with_preds)

cv2.destroyAllWindows()
print("Tüm görseller işlendi.")