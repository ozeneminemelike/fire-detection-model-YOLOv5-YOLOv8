import torch
import cv2
import os

model = torch.hub.load(
    '/yolov5',  # yolov5 file path
    'custom',
    path='/best.pt', #model file path
    source='local'
)

input_folder = "/test_images"     # Test pics file path
output_folder = "/test_outputs"   # Output pics file path
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"{image_name} okunamadı, atlandı.")
        continue

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    results.render()
    result_img = results.ims[0]
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

    # show pics on the screen(Comment if you won't use it)
    cv2.imshow("Tahmin", result_img)
    cv2.waitKey(500)

    output_path = os.path.join(output_folder, f"pred_{image_name}")
    cv2.imwrite(output_path, result_img)

cv2.destroyAllWindows()
print("Tüm test görselleri işlendi ve kaydedildi.")
