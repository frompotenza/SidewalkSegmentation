import torch
import numpy as np
import cv2
import os


folder_path = r"prediction_pipeline\output_edges"
os.makedirs(folder_path, exist_ok=True)
file_index = 1

height_delta = 50  # height in pixels
dx_distance_threshold = 200
distance_calculations = 4


def torch_mask_to_numpy(mask_tensor):
    """
    Converts a PyTorch mask tensor to a 2D binary NumPy array for processing.
    Assumes input mask is [H, W] or [1, H, W] or [N, H, W] (single mask at a time).
    """
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor[0]  # Take first if batched
    mask_np = mask_tensor.detach().cpu().numpy()
    mask_bin = (mask_np > 0.5).astype(np.uint8)  # Threshold to binary mask
    return mask_bin


def get_movement_recommendation(mask_tensor):
    global file_index

    mask_bin = torch_mask_to_numpy(mask_tensor)

    # Get edges (better than raw mask for line detection)
    # Scale by 255 because binary has values too small to be detected as edge
    # Canny uses colour scale
    img_edges = cv2.Canny(mask_bin * 255, 10, 150)

    height, width = img_edges.shape[:2]
    # Coordinates: center X, bottom Y
    x_center = width // 2
    target_y = height - 1

    left_recommendation_count = 0
    straight_recommendation_count = 0
    right_recommendation_count = 0

    for _ in range(distance_calculations):
        target_y -= height_delta
        row_values = img_edges[target_y, :]  # Get the full row

        # left distance
        for dx in range(1, x_center + 1):
            if row_values[x_center - dx] != 0:
                if dx > dx_distance_threshold:
                    straight_recommendation_count += 0.5
                else:
                    right_recommendation_count += (
                        1  # too close to the left --> go right
                    )
                break

        # right distance
        for dx in range(1, width - x_center):
            if row_values[x_center + dx] != 0:
                if dx > dx_distance_threshold:
                    straight_recommendation_count += 0.5
                else:
                    left_recommendation_count += 1  # too close to the right --> go left
                break

    final_recommendation = "straight"
    if 0 < left_recommendation_count >= straight_recommendation_count:
        final_recommendation = "left"
    elif 0 < right_recommendation_count >= straight_recommendation_count:
        final_recommendation = "right"

    print(
        f"edges_{file_index} - right = {right_recommendation_count}, left = {left_recommendation_count}, straight = {straight_recommendation_count}"
    )

    frame = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)

    # Draw direction arrow
    frame = draw_direction_arrow(frame, final_recommendation)

    # Save to disk
    file_name = f"edges_{file_index}.png"
    full_path = os.path.join(folder_path, file_name)
    cv2.imwrite(full_path, frame)
    file_index += 1

    return final_recommendation


def draw_direction_arrow(frame, direction):
    height, width = frame.shape[:2]
    center = (width // 2, height - 50)  # Arrow base near the bottom center
    length = 100  # Length of the arrow

    if direction == "left":
        end_point = (center[0] - length, center[1] - 50)
        color = (0, 255, 0)  # Green
    elif direction == "right":
        end_point = (center[0] + length, center[1] - 50)
        color = (255, 0, 0)  # Blue
    elif direction == "straight":
        end_point = (center[0], center[1] - length)
        color = (0, 0, 255)  # Red
    else:
        raise ValueError("Direction must be 'left', 'right', or 'straight'")

    # Draw the arrow
    cv2.arrowedLine(frame, center, end_point, color=color, thickness=5, tipLength=0.4)

    return frame
