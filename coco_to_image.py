import json
import os
import numpy as np
import cv2
from pycocotools import mask as mask_utils


def coco_to_image(json_path, output_path, draw_bbox=True):
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    image_info = coco_data["images"][0]
    height, width = image_info["height"], image_info["width"]
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for ann in coco_data["annotations"]:
        seg = ann["segmentation"]

        if isinstance(seg, dict):  # RLE
            mask = mask_utils.decode(seg)
        elif isinstance(seg, list):  # Polygon
            mask = np.zeros((height, width), dtype=np.uint8)
            for poly in seg:
                pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        else:
            raise ValueError("Unsupported segmentation format")

        color = (0, 255, 0)  # Green
        image[mask == 1] = color

        if draw_bbox:
            x, y, w, h = ann["bbox"]
            cv2.rectangle(
                image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2
            )

    cv2.imwrite(output_path, image)
    print(f"✅ Saved image to: {output_path}")


# ---- Example usage ----
json_path = "segmentation_outputs/142_jpg.rf.c04b5fd1f1fdbd25a96195a8287157b0.json"  # Replace with your actual JSON file
output_path = "output_mask.png"  # Desired output image file

coco_to_image(json_path, output_path)
