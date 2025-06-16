import cv2
import os
from ultralytics import YOLO

model = YOLO(r"prediction_pipeline\best_model_8.pt")

video_path = r"prediction_pipeline\WhatsApp Video 2025-06-16 at 15.37.43_5ffd0d83.mp4"
output_folder = r"prediction_pipeline\predicted_frames"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 0.25)  # Number of frames to skip for 0.5s interval

confidence_threshold = 0.9
label_count_threshold = 3
frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Show current frame regardless of prediction
    cv2.imshow("Video Playback", frame)

    # Predict only every frame_interval frames
    if frame_count % frame_interval == 0:
        # Resize frame to 640x640 before inference
        resized_frame = cv2.resize(frame, (640, 640))

        # Run YOLO prediction
        results = model(resized_frame)[0]

        within_confidence = True

        for box in results.boxes:
            confidence = float(box.conf[0])  # Confidence score (0.0 - 1.0)

            if confidence < confidence_threshold:
                within_confidence = False

        if not within_confidence or len(results.boxes) > label_count_threshold:
            continue

        # Draw predictions on frame
        annotated_frame = results.plot()

        # Show prediction result
        cv2.imshow("YOLO Prediction", annotated_frame)

        # Save prediction frame
        save_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(save_path, annotated_frame)
        print(f"Saved {save_path}")
        saved_count += 1

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
