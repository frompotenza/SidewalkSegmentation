import cv2
import os
from ultralytics import YOLO
from directions_calculation import get_movement_recommendation

# Load YOLO model
model = YOLO(r"prediction_pipeline\best_model_8.pt")

# Set paths
video_path = r"prediction_pipeline\video2.mp4"
output_folder = r"prediction_pipeline\predicted_frames"
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video.")
    exit()

# Determine frame skip interval (e.g., every 0.25s)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 0.25)

# YOLO and filtering parameters
confidence_threshold = 0.9
max_detections = 1
bbox_size_threshold = 6000  # Minimum bounding box area

frame_count = 0
saved_count = 0


while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Display original video stream
    cv2.imshow("Video Playback", frame)

    # Perform detection at defined interval
    if frame_count % frame_interval == 0:
        # Resize frame to 640x640 before inference
        resized_frame = cv2.resize(frame, (640, 640))

        # Run YOLO prediction
        results = model(
            resized_frame, conf=confidence_threshold, max_det=max_detections
        )[0]

        if results.boxes is None or results.masks is None:
            frame_count += 1
            continue

        valid_detection = False

        # Only one detection (max_detections=1)
        bbox = results.boxes[0]
        mask = results.masks[0]
        _, _, width, height = bbox.xywh[0]
        # Filter out predictions that are too small
        if width * height >= bbox_size_threshold:
            valid_detection = True
            recommendation = get_movement_recommendation(mask.data)
        else:
            frame_count += 1
            continue

        # Draw prediction results on frame
        annotated_frame = results.plot()
        cv2.imshow("YOLO Prediction", annotated_frame)

        # Save prediction frame
        save_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(save_path, annotated_frame)
        print(f"Saved {save_path}")
        saved_count += 1

    frame_count += 1

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
