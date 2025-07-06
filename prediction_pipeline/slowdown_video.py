import cv2
import os

# Input and output paths
input_path = r"Test_Videos_Sidewalk\segmented_only_output.mp4"
output_path = r"prediction_pipeline\video2_slow_0.25x.mp4"

# Open the input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Get original properties
original_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set slowed-down FPS (0.25x speed)
slow_fps = original_fps * 0.25  # 4x slower

# Define codec and create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, slow_fps, (width, height))

# Copy all frames into the slower video
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    frame_count += 1

# Release resources
cap.release()
out.release()

print(f"Slowed video saved to: {output_path}")
print(f"Original FPS: {original_fps:.2f}, Slowed FPS: {slow_fps:.2f}")
print(f"Total frames written: {frame_count}")
