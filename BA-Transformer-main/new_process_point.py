# import cv2
# import os
# import random
# import torch
# import numpy as np
# import skimage.draw
# from tqdm import tqdm
# import torch.nn.functional as F

# # -------------------------
# # NEW UTILITY: Resize and clip an image or mask to 512x512.
# def resize_and_clip(img, target_size=(512, 512)):
#     # Resize from 1024x1024 to 512x512 (use INTER_NEAREST for masks)
#     resized = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
#     # Clip the values to [0, 255]
#     resized = np.clip(resized, 0, 255)
#     return resized

# # -------------------------
# def create_circular_mask(h, w, center, radius):
#     Y, X = np.ogrid[:h, :w]
#     dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
#     mask = dist_from_center <= radius
#     return mask

# # -------------------------
# def NMS(heatmap, kernel=13):
#     hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=(kernel - 1) // 2)
#     keep = (hmax == heatmap).float()
#     return heatmap * keep, hmax, keep

# # -------------------------
# def draw_msra_gaussian(heatmap, center, sigma):
#     tmp_size = sigma * 3
#     mu_x = int(center[0] + 0.5)
#     mu_y = int(center[1] + 0.5)
#     h, w = heatmap.shape[0], heatmap.shape[1]
#     ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
#     br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
#     if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
#         return heatmap
#     size = 2 * tmp_size + 1
#     x = np.arange(0, size, 1, np.float32)
#     y = x[:, np.newaxis]
#     x0 = y0 = size // 2
#     g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
#     # Calculate the bounds for the gaussian and for the image
#     g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
#     g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
#     img_x = max(0, ul[0]), min(br[0], w)
#     img_y = max(0, ul[1]), min(br[1], h)
#     heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
#         heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], 
#         g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
#     )
#     return heatmap

# # -------------------------
# def kpm_gen(mask_path, R, N):
#     """
#     Generate key-patch map from a given mask.
#     For pneumothorax segmentation, we use modified thresholds:
#       - R: Base radius for drawing keypoints (used indirectly here).
#       - N: Neighborhood size for filtering keypoints.
#     """
#     # Read mask from .png, ensure grayscale
#     mask_orig = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     if mask_orig is None:
#         raise ValueError(f"Cannot read mask from {mask_path}")
    
#     # Resize the mask to 512x512 and clip values.
#     mask_orig = resize_and_clip(mask_orig, (512, 512))
    
#     # Create a copy for processing contours.
#     mask_proc = mask_orig.copy()
    
#     # Since the original data is uint8 (0-255), threshold it to get binary mask
#     # (for instance: pixels > 127 are considered foreground).
#     _, mask_bin = cv2.threshold(mask_proc, 127, 255, cv2.THRESH_BINARY)
    
#     # Downsample the mask for contour extraction.
#     # Here we choose a factor of 2 instead of 4 since the task and resolution differ.
#     mask_small = cv2.resize(mask_bin, (mask_bin.shape[1] // 2, mask_bin.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    
#     # Convert to uint8 if needed
#     mask_small = np.uint8(mask_small)
    
#     contours, hierarchy = cv2.findContours(mask_small, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#     num_contours = len(contours)
    
#     # Optional: draw contours on a color copy for debugging (if needed)
#     mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
#     draw_mask = cv2.drawContours(mask_color.copy(), contours, -1, (0, 0, 255), 1)
    
#     # We generate the key-patch heatmap in full resolution: 512x512.
#     point_heatmap = np.zeros((512, 512), dtype=np.float32)
    
#     # Process each contour
#     for contour in contours:
#         points = contour[:, 0]  # Each contour point is (x,y)
#         # Scale points back to the original 512x512 resolution
#         points = points * 2  # because we downsampled by factor 2
#         points_number = points.shape[0]
#         if points_number < 20:  # Skip very small contours
#             continue

#         # For pneumothorax segmentation, adjust the radius and neighborhood settings.
#         # You can adjust these values as you see fit. Here we use R and N as base parameters.
#         if points_number < 150:
#             radius = R  # for example, R could be set to 8
#             neighbor_points_n_oneside = N  # e.g. N could be 5
#         elif points_number < 300:
#             radius = R + 4  # for a slightly larger spread
#             neighbor_points_n_oneside = N + 5
#         else:
#             radius = R + 7
#             neighbor_points_n_oneside = N + 10

#         # Compute the "overlap area" for each point.
#         stds = []
#         for i in range(points_number):
#             pt = points[i]
#             mask_circ = create_circular_mask(512, 512, (pt[0], pt[1]), radius)
#             # Overlap is computed with the original (binary) mask.
#             overlap = np.sum(mask_circ * (mask_orig > 0)) / (np.pi * radius * radius)
#             stds.append(overlap)
        
#         stds = np.array(stds)
#         # Select keypoints based on neighborhood comparison.
#         selected_points = []
#         for i in range(points_number):
#             neighbor_idx = np.concatenate([
#                 np.arange(-neighbor_points_n_oneside, 0),
#                 np.arange(1, neighbor_points_n_oneside + 1)
#             ]) + i
#             # Handle wrap-around
#             neighbor_idx[neighbor_idx < 0] += points_number
#             neighbor_idx[neighbor_idx >= points_number] -= points_number
            
#             if stds[i] < np.min(stds[neighbor_idx]) or stds[i] > np.max(stds[neighbor_idx]):
#                 point_heatmap = draw_msra_gaussian(point_heatmap, (points[i, 0], points[i, 1]), sigma=R)
#                 selected_points.append(points[i])
                
#         # Optionally, you could compute IOU here between a polygon of selected points and mask_orig
#         # to verify how representative the keypoints are.

#     return mask_orig, point_heatmap

# # -------------------------
# def process_dataset():
#     # Directories for the pneumothorax dataset.
#     mask_dir = r"C:/Users/nisha/Desktop/Research Project May-June/mask"
#     output_dir = r"C:/Users/nisha/Desktop/Research Project May-June/gt_keypatch"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # List all .png mask files.
#     mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
#     mask_files.sort()
    
#     # Set R and N parameters for your task.
#     # For example, you might set R (base radius for Gaussians) to 8 and N (neighbor window half-size) to 5.
#     R = 8
#     N = 5
    
#     for mask_file in tqdm(mask_files):
#         mask_path = os.path.join(mask_dir, mask_file)
#         # Generate the key-patch (point) heatmap.
#         original_mask, point_heatmap = kpm_gen(mask_path, R, N)
        
#         # Save the generated heatmap as .npy file (or you can save as .png if preferred)
#         output_name = os.path.splitext(mask_file)[0] + '.npy'
#         output_path = os.path.join(output_dir, output_name)
#         np.save(output_path, point_heatmap)

# if __name__ == '__main__':
#     process_dataset()


import cv2
import os
import random
import torch
import numpy as np
import skimage.draw
from tqdm import tqdm
import torch.nn.functional as F

# -------------------------
# Utility function: Resize and clip an image or mask to 512x512.
def resize_and_clip(img, target_size=(512, 512)):
    # Resize from 1024x1024 to 512x512 (use INTER_NEAREST for masks)
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
    # Clip the values to [0, 255]
    resized = np.clip(resized, 0, 255)
    return resized

# -------------------------
def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

# -------------------------
def NMS(heatmap, kernel=13):
    hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heatmap).float()
    return heatmap * keep, hmax, keep

# -------------------------
def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    h, w = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    # Calculate the bounds for the gaussian and for the image
    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], 
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    )
    return heatmap

# -------------------------
def kpm_gen(mask_path, R, N):
    """
    Generate key-patch map from a given mask.
    For pneumothorax segmentation, we use modified thresholds:
      - R: Base radius for drawing keypoints (used indirectly here).
      - N: Neighborhood size for filtering keypoints.
    """
    # Read mask from .png, ensure grayscale
    mask_orig = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_orig is None:
        raise ValueError(f"Cannot read mask from {mask_path}")
    
    # Resize the mask to 512x512 and clip values.
    mask_orig = resize_and_clip(mask_orig, (512, 512))
    
    # Create a copy for processing contours.
    mask_proc = mask_orig.copy()
    
    # Threshold mask to binary (e.g., pixels > 127 are foreground)
    _, mask_bin = cv2.threshold(mask_proc, 127, 255, cv2.THRESH_BINARY)
    
    # Downsample the mask for contour extraction.
    # Here we choose a factor of 2 since the input is now 512x512.
    mask_small = cv2.resize(mask_bin, (mask_bin.shape[1] // 2, mask_bin.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    mask_small = np.uint8(mask_small)
    
    contours, hierarchy = cv2.findContours(mask_small, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Initialize heatmap and track selected keypoints count.
    point_heatmap = np.zeros((512, 512), dtype=np.float32)
    total_selected_points = 0
    
    for contour in contours:
        points = contour[:, 0]  # Each point in contour is (x, y)
        # Scale points back to the original 512x512 resolution
        points = points * 2  # because we downsampled by factor 2
        points_number = points.shape[0]
        if points_number < 20:  # Skip very small contours
            continue

        # Adjust parameters for pneumothorax segmentation.
        if points_number < 150:
            radius = R            # e.g., R could be set to 8
            neighbor_points_n_oneside = N   # e.g., N could be 5
        elif points_number < 300:
            radius = R + 4
            neighbor_points_n_oneside = N + 5
        else:
            radius = R + 7
            neighbor_points_n_oneside = N + 10

        stds = []
        for i in range(points_number):
            pt = points[i]
            mask_circ = create_circular_mask(512, 512, (pt[0], pt[1]), radius)
            overlap = np.sum(mask_circ * (mask_orig > 0)) / (np.pi * radius * radius)
            stds.append(overlap)
        
        stds = np.array(stds)
        # Select keypoints based on neighborhood comparison.
        for i in range(points_number):
            neighbor_idx = np.concatenate([
                np.arange(-neighbor_points_n_oneside, 0),
                np.arange(1, neighbor_points_n_oneside + 1)
            ]) + i
            # Handle circular (wrap-around) indexing.
            neighbor_idx[neighbor_idx < 0] += points_number
            neighbor_idx[neighbor_idx >= points_number] -= points_number
            
            if stds[i] < np.min(stds[neighbor_idx]) or stds[i] > np.max(stds[neighbor_idx]):
                point_heatmap = draw_msra_gaussian(point_heatmap, (points[i, 0], points[i, 1]), sigma=R)
                total_selected_points += 1

    return mask_orig, point_heatmap, total_selected_points

# -------------------------
def process_dataset():
    # Directories for the pneumothorax dataset.
    mask_dir = r"C:/Users/nisha/Desktop/Research Project May-June/mask"
    output_dir = r"C:/Users/nisha/Desktop/Research Project May-June/gt_keypatch"
    os.makedirs(output_dir, exist_ok=True)
    
    # List all .png mask files.
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    mask_files.sort()
    
    # Set R and N parameters for this task.
    R = 8
    N = 5

    # Use tqdm to display a progress bar.
    pbar = tqdm(mask_files, desc="Processing Masks", unit="file")
    for mask_file in pbar:
        mask_path = os.path.join(mask_dir, mask_file)
        original_mask, point_heatmap, keypt_count = kpm_gen(mask_path, R, N)
        
        # Save the generated key-patch heatmap as .npy file.
        output_name = os.path.splitext(mask_file)[0] + '.npy'
        output_path = os.path.join(output_dir, output_name)
        np.save(output_path, point_heatmap)
        
        # Update the tqdm description with the number of keypoints found.
        pbar.set_postfix({"keypoints": keypt_count})
        # Optionally, print or log detailed progress info:
        print(f"Processed {mask_file}: {keypt_count} keypoints generated.")

if __name__ == '__main__':
    process_dataset()
