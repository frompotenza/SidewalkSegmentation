import os
from ultralytics import YOLO


def main():
    # Change these paths as needed
    dataset_dir = "yolo_dataset"
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")

    # Check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    # Load YOLOv8 segmentation model (pretrained)
    model = YOLO(
        "yolov8l-seg.pt"
    )  # Change to yolov8n-seg.pt or yolov8m-seg.pt if preferred

    # Train
    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device="0",  # Use 'cpu' if no GPU available
        name="yolo11l-seg segmentation experiment",
    )


if __name__ == "__main__":
    main()
