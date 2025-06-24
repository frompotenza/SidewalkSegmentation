import torch
import numpy as np
import cv2
from sklearn.cluster import KMeans


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


def get_hough_line_directions(mask_tensor, max_lines=3):
    mask_bin = torch_mask_to_numpy(mask_tensor)

    # Get edges (better than raw mask for line detection)
    # Scale by 255 because binary has values too small to be detected as edge
    # Canny uses colour scale
    edges = cv2.Canny(mask_bin * 255, 10, 150)

    # Hough Line Transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.rad2deg(theta)
            angle_deg = angle_deg % 180  # Normalize to [0, 180)
            angles.append(angle_deg)

    if len(angles) > 0:
        angles_np = np.array(angles).reshape(-1, 1)
        for k in [1, 2, 3]:
            kmeans = KMeans(n_clusters=k, n_init="auto").fit(angles_np)
            centers = kmeans.cluster_centers_.flatten()
            print(f"{k} direction(s):", centers)

    return angles
