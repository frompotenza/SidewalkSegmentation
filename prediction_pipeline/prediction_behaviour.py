import cv2
import os
from ultralytics import YOLO
from directions_calculation import get_movement_recommendation

# Load YOLO model
model = YOLO(r"prediction_pipeline\best_model_8.pt")

# Set paths
video_path = r"Test_Videos_Sidewalk\WhatsApp Video 2025-07-06 at 15.33.12_f87fea2e.mp4"
output_video_path = r"Test_Videos_Sidewalk\segmented_only_output.mp4"
output_folder = r"Test_Videos_Sidewalk\predicted_frames"
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video.")
    exit()

# Get original video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_interval = int(fps * 0.25)  # Analyze every 0.25s

# Prepare video writer (only initialized after first valid frame)
video_writer = None

# Parameters
confidence_threshold = 0.9
max_detections = 1
bbox_size_threshold = 150000  # Minimum bbox area

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    if frame_count % frame_interval == 0:
        # Resize for YOLO input
        resized_frame = cv2.resize(frame, (640, 640))

        # Run detection
        results = model(
            resized_frame, conf=confidence_threshold, max_det=max_detections
        )[0]

        if (
            results.boxes is not None
            and results.masks is not None
            and len(results.boxes) > 0
        ):
            bbox = results.boxes[0]
            mask = results.masks[0]
            _, _, width, height = bbox.xywh[0]

            if width * height >= bbox_size_threshold:
                # Movement recommendation (optional logic)
                get_movement_recommendation(mask.data)

                # Overlay prediction
                segmented_resized = results.plot()

                # Resize back to original dimensions
                segmented_frame = cv2.resize(
                    segmented_resized, (frame_width, frame_height)
                )

                # Initialize video writer on first valid frame
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        output_video_path, fourcc, fps, (frame_width, frame_height)
                    )

                # Write segmented frame
                video_writer.write(segmented_frame)

                # Save frame image
                save_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(save_path, segmented_frame)
                print(f"Saved {save_path}")
                saved_count += 1

    frame_count += 1

# Cleanup
cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()

print(f"Segmented video saved: {output_video_path}")
