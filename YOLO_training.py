import os
import json
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
from pycocotools import mask as maskUtils
import cv2

# --- UTILS ---


def coco_poly_to_yolo_segmentation(segmentation, img_w, img_h):
    """
    Converts COCO polygon segmentation (list of lists) to YOLO segmentation format
    normalized to [0,1] relative coordinates (x, y).
    Input:
      segmentation: list of list of floats: [[x1,y1,x2,y2,...]]
      img_w: image width
      img_h: image height
    Output:
      list of floats: [x1, y1, x2, y2, ...] normalized (0-1)
    """
    # COCO segmentation is usually a list of polygons, we use the first polygon here
    poly = (
        segmentation[0]
        if isinstance(segmentation, list) and len(segmentation) > 0
        else []
    )
    yolo_seg = []
    for i in range(0, len(poly), 2):
        x_norm = poly[i] / img_w
        y_norm = poly[i + 1] / img_h
        yolo_seg.extend([x_norm, y_norm])
    return yolo_seg


def rle_to_polygon(rle):
    """
    Convert RLE to polygon format.
    Returns a list of polygons (each polygon is a list of x,y points)
    """
    if isinstance(rle["counts"], list):
        # Uncompressed RLE
        rle_obj = maskUtils.frPyObjects(rle, rle["size"][0], rle["size"][1])
    else:
        # Compressed RLE
        rle_obj = rle

    # Decode RLE mask to binary mask
    mask = maskUtils.decode(rle_obj)

    # Find contours (polygons) from binary mask using OpenCV
    contours, _ = cv2.findContours(
        mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) >= 6:  # valid polygon has at least 3 points (x,y pairs)
            polygons.append(contour)
    return polygons


def convert_coco_to_yolo_labels(coco_json_path, yolo_label_path):
    """
    Convert COCO annotations to YOLO segmentation label file.
    Assumes category_id 1 = sidewalk, which maps to class 0 in YOLO.
    Saves one line per annotation:
      <class_id> <segmentation points normalized>
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    # We expect exactly one image entry
    img_info = coco["images"][0]
    img_w, img_h = img_info["width"], img_info["height"]

    lines = []
    for ann in coco["annotations"]:
        if ann["category_id"] != 1:
            continue  # skip other categories if any
        seg = ann["segmentation"]

        if isinstance(seg, list):
            # Polygon segmentation
            yolo_seg = coco_poly_to_yolo_segmentation(seg, img_w, img_h)
        else:
            # RLE segmentation, decode to polygons
            polygons = rle_to_polygon(seg)
            if not polygons:
                print(f"Warning: No polygons found from RLE in {coco_json_path.name}")
                return False
            # Use first polygon only for YOLO format
            yolo_seg = coco_poly_to_yolo_segmentation([polygons[0]], img_w, img_h)

        # Class 0 because only one category: sidewalk
        cls_id = 0
        line = f"{cls_id} " + " ".join(f"{x:.6f}" for x in yolo_seg)
        lines.append(line)

    if not lines:
        print(f"Warning: No valid annotations found in {coco_json_path.name}")
        return False

    with open(yolo_label_path, "w") as f:
        f.write("\n".join(lines))
    return True


# --- MAIN PROCESS ---
if __name__ == "__main__":

    # --- SETTINGS ---
    DATA_ROOT = Path(
        "segmentation_outputs"
    ).resolve()  # Your input folder with images + jsons
    YOLO_DIR = Path("yolo_data").resolve()  # Output dataset folder for YOLO format
    IMG_EXT = ".jpg"  # Your image extension
    EPOCHS = 50
    IMG_SIZE = 640
    SEED = 42

    # --- PREPARE YOLO FOLDERS ---
    for folder in [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ]:
        (YOLO_DIR / folder).mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    # Gather all image files with matching json files
    all_images = []
    for file in DATA_ROOT.iterdir():
        if file.suffix.lower() == IMG_EXT:
            json_file = file.with_suffix(".json")
            if json_file.exists():
                all_images.append(file)

    print(f"Found {len(all_images)} images with matching JSON files")

    # Shuffle and split dataset
    random.shuffle(all_images)
    n = len(all_images)
    train_end = int(0.7 * n)
    val_end = train_end + int(0.2 * n)

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:],
    }

    print(
        f"Split into train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )

    # Copy images and convert labels
    for split, files in splits.items():
        for img_path in files:
            json_path = img_path.with_suffix(".json")

            # Copy image
            dest_img = YOLO_DIR / f"images/{split}" / img_path.name
            shutil.copyfile(img_path, dest_img)

            # Convert and save label
            dest_label = YOLO_DIR / f"labels/{split}" / (img_path.stem + ".txt")
            success = convert_coco_to_yolo_labels(json_path, dest_label)
            if not success:
                # Remove copied image if label failed
                dest_img.unlink()
                print(f"Skipped {img_path.name} due to missing or invalid annotation")

    # --- WRITE DATA YAML ---

    data_yaml = f"""\
path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val
names:
- sidewalk
"""

    with open(YOLO_DIR / "data.yaml", "w") as f:
        f.write(data_yaml)

    print(f"Written data.yaml to {YOLO_DIR / 'data.yaml'}")

    # --- TRAIN YOLOv8 SEGMENTATION MODEL ---

    model = YOLO("yolov8s-seg.pt")  # small segmentation model pretrained

    print("Starting training...")
    model.train(data=str(YOLO_DIR / "data.yaml"), epochs=EPOCHS, imgsz=IMG_SIZE)
    print("Training finished!")
