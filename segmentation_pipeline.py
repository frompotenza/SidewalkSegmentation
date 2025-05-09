import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from segment_anything import sam_model_registry, SamPredictor
from pycocotools import mask as mask_utils

# --- Configuration ---
image_folder = "<IMAGE_FOLDER>"
checkpoint_path = "PATH_TO_MODEL"
model_type = "vit_h"
output_root = "segmentation_outputs"
os.makedirs(output_root, exist_ok=True)

# --- Load SAM model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)


# --- Helper to convert binary mask to COCO RLE ---
def binary_mask_to_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")  # bytes to str for JSON
    return rle


annotation_id = 1
image_id = 1

# --- Process each image ---
for fname in tqdm(os.listdir(image_folder)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(image_folder, fname)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # Set up predictor
    predictor.set_image(image_rgb)
    points = []
    labels = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            labels.append(1)
            cv2.circle(image_bgr, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Click points, press 's' to segment", image_bgr)

    print(f"\nProcessing image: {fname}")
    cv2.imshow("Click points, press 's' to segment", image_bgr)
    cv2.setMouseCallback("Click points, press 's' to segment", click_event)

    while True:
        key = cv2.waitKey(1)
        if key == ord("s") and points:
            break
        elif key == 27:  # ESC to exit
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

    # Predict mask from clicks
    input_points = np.array(points)
    input_labels = np.array(labels)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )
    mask = masks[0]

    # Overlay the mask onto the original image
    overlay = image_bgr.copy()
    overlay[mask] = [0, 255, 0]  # Green for segmented area

    # Save the overlay image
    base_name = os.path.splitext(fname)[0]
    overlay_path = os.path.join(output_root, f"{base_name}_segmented.png")
    cv2.imwrite(overlay_path, overlay)

    # --- Save individual COCO annotation JSON ---
    coco_data = {
        "info": {
            "description": "Manual SAM Segmentation",
            "date_created": datetime.now().isoformat(),
        },
        "images": [{"id": image_id, "file_name": fname, "width": w, "height": h}],
        "annotations": [
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": binary_mask_to_rle(mask),
                "area": int(mask.sum()),
                "bbox": list(cv2.boundingRect(mask.astype(np.uint8))),
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "object"}],
    }

    json_path = os.path.join(output_root, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    annotation_id += 1
    image_id += 1

print("\n✅ All segmentations complete. JSON and segmented images saved.")
