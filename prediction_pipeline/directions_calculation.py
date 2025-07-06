import torch
import numpy as np
import cv2
import os
import time
import pyttsx3
from collections import Counter


# Output folder for edge maps with direction arrows
folder_path = r"prediction_pipeline\output_edges"
os.makedirs(folder_path, exist_ok=True)


# Tuning parameters for direction logic
height_delta = 15
dx_distance_threshold = 200
steps_distance = 12
weight_decay = 0.05
delta_threshold = 5  # Reset direction history after 5 seconds

# TTS parameters
engine = pyttsx3.init()
engine.setProperty(
    "voice",
    r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_DE-DE_HEDDA_11.0",
)

# Global state
last_three_predictions = []
obstacle_count_sequence = 0
obstacle_threshold = 3
last_call_time = None
file_index = 1
last_recommendation = None


def torch_mask_to_numpy(mask_tensor):
    """Convert a PyTorch mask tensor to binary 2D NumPy array."""
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor[0]
    mask_np = mask_tensor.detach().cpu().numpy()
    return (mask_np > 0.5).astype(np.uint8)


def get_movement_recommendation(mask_tensor):
    """
    Analyze binary edge mask to determine direction.
    Direction is based on distance from center to edges at multiple row heights.
    """
    global file_index, last_three_predictions, last_call_time, last_recommendation, obstacle_count_sequence

    mask_bin = torch_mask_to_numpy(mask_tensor)

    # Get edges (better than raw mask for line detection)
    # Scale by 255 because binary has values too small to be detected as edge
    # Canny uses colour scale
    img_edges = cv2.Canny(mask_bin * 255, 10, 150)

    height, width = img_edges.shape[:2]
    x_center = width // 2
    target_y = height - 1

    left_score = 0
    straight_score = 0
    right_score = 0
    current_weight = 1

    # Analyze rows above the bottom in steps
    for _ in range(steps_distance):
        target_y -= height_delta
        if target_y < 0:
            break

        row_values = img_edges[target_y, :]

        # Check edge distance on left side
        for dx in range(1, x_center + 1):
            if row_values[x_center - dx] != 0:
                if dx > dx_distance_threshold:
                    straight_score += current_weight / 2
                else:
                    right_score += current_weight  # too close to the left --> go right
                break

        # Check edge distance on right side
        for dx in range(1, width - x_center):
            if row_values[x_center + dx] != 0:
                if dx > dx_distance_threshold:
                    straight_score += current_weight / 2
                else:
                    left_score += current_weight  # too close to the right --> go left
                break

        current_weight -= weight_decay

    column_values = img_edges[:, x_center]
    if any(x != 0 for x in column_values[-height_delta * steps_distance :]):
        obstacle_count_sequence = min(obstacle_count_sequence + 1, obstacle_threshold)
    else:
        obstacle_count_sequence = max(obstacle_count_sequence - 1, 0)

    # Recommended direction based on weighted scores
    if 0 < left_score >= straight_score:
        final_recommendation = "links"
    elif 0 < right_score >= straight_score:
        final_recommendation = "rechts"
    else:
        final_recommendation = "geradeaus"

    # Reset direction history if too much time has passed
    current_time = time.perf_counter()
    if last_call_time is not None and current_time - last_call_time > delta_threshold:
        print("\n\n!!! Resetting recent predictions !!!\n\n")
        last_three_predictions = []
    last_call_time = current_time

    # Keep last 3 predictions to stabilize output
    last_three_predictions.append(final_recommendation)
    if len(last_three_predictions) > 3:
        last_three_predictions.pop(0)
        final_recommendation, _ = Counter(last_three_predictions).most_common(1)[0]

    # ~~(Un)comment for TTS recommendation~~
    if final_recommendation != last_recommendation:
        engine.say(f"{final_recommendation}")
    if obstacle_count_sequence >= obstacle_threshold:
        engine.say("Obstacle ahead.")
        obstacle_count_sequence = 0
    engine.runAndWait()
    last_recommendation = final_recommendation

    # Save visualization with arrow
    frame = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
    frame = draw_direction_arrow(frame, final_recommendation)

    # Save visualization
    save_path = os.path.join(folder_path, f"edges_{file_index}.png")
    cv2.imwrite(save_path, frame)
    file_index += 1

    return final_recommendation


def draw_direction_arrow(frame, direction):
    """
    Draws a colored arrow (green=left, blue=right, red=straight) on the image
    """
    height, width = frame.shape[:2]
    center = (width // 2, height - 50)  # Arrow base near the bottom center
    length = 100  # Length of the arrow

    if direction == "links":
        end_point = (center[0] - length, center[1] - 50)
        color = (0, 255, 0)  # Green
    elif direction == "rechts":
        end_point = (center[0] + length, center[1] - 50)
        color = (255, 0, 0)  # Blue
    elif direction == "geradeaus":
        end_point = (center[0], center[1] - length)
        color = (0, 0, 255)  # Red
    else:
        raise ValueError("Direction must be 'left', 'right', or 'straight'")

    cv2.arrowedLine(frame, center, end_point, color=color, thickness=5, tipLength=0.4)
    return frame
