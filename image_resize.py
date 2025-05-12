import os
import cv2

input_folder = "Dataset"
output_folder = "Dataset_resized"
target_size = (640, 640)  # Width, Height

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            resized = cv2.resize(img, target_size)
            cv2.imwrite(os.path.join(output_folder, filename), resized)
            print(f"Saved: {filename}")
