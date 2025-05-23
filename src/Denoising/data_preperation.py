import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def discard_cells(crop_dir, output_dir):
    """
    Discards cells where the ratio between the height and width is greater than 2.
    Saves the remaining cells in a new directory.

    Parameters:
    -----------
    crop_dir : str
        Path to the directory containing cropped cell images.
    output_dir : str
        Path where the filtered cell images will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Gather crop files
    crop_files = sorted([f for f in os.listdir(crop_dir) if f.endswith(('.png', '.jpg'))])
    print(f"Found {len(crop_files)} cropped cell(s) in {crop_dir}...")

    for crop_file in crop_files:
        crop_path = os.path.join(crop_dir, crop_file)
        img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Could not read {crop_file} — skipping.")
            continue

        h, w = img.shape[:2]
        if h > 1.5 * w or w > 1.5 * h:
            print(f"[INFO] Discarding {crop_file} due to aspect ratio.")
            continue
        elif h <= 75 or w <= 75:
            print(f"[INFO] Discarding {crop_file} due to small dimensions.")
            continue

        # Save the valid cell image
        output_path = os.path.join(output_dir, crop_file)
        cv2.imwrite(output_path, img)

def resize_cells(filtered_cells_dir, output_dir, target_size=(128, 128)):
    """
    Resizes cropped cell images to a target size and saves them in a new directory.

    Parameters:
    -----------
    filtered_cells_dir : str
        Path to the directory containing cropped cell images.
    output_dir : str
        Path where the resized cell images will be saved.
    target_size : tuple
        Target size for resizing (width, height).
    """

    os.makedirs(output_dir, exist_ok=True)

    # Gather crop files
    crop_files = sorted([f for f in os.listdir(filtered_cells_dir) if f.endswith(('.png', '.jpg'))])
    print(f"Found {len(crop_files)} cropped cell(s) in {filtered_cells_dir}...")

    for crop_file in crop_files:
        cell_path = os.path.join(filtered_cells_dir, crop_file)
        img = cv2.imread(cell_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Could not read {crop_file} — skipping.")
            continue

        target_width, target_height = target_size
        original_height, original_width = img.shape[:2]

        # Compute resize ratios
        height_change_req = original_height - target_height
        width_change_req = original_width - target_width

        # Determine dominant resizing direction (height or width)
        if max(np.abs(height_change_req), np.abs(width_change_req)) == np.abs(height_change_req):
            dominant_dim = height_change_req
        else:
            dominant_dim = width_change_req

        # Use INTER_AREA if dominant_dim > 0 (downscaling), otherwise INTER_LANCZOS4 (upscaling)
        interpolation = cv2.INTER_AREA if dominant_dim > 0 else cv2.INTER_LANCZOS4

        # Resize (OpenCV uses (width, height) order!)
        resized_img = cv2.resize(img, (target_width, target_height), interpolation=interpolation)

        # Save the resized image
        output_path = os.path.join(output_dir, crop_file)
        cv2.imwrite(output_path, resized_img)

def plot_histogram_of_cell_dimensions(crop_dir):
    """
    Plots a histogram of the dimensions of cropped cell images.

    Parameters:
    -----------
    crop_dir : str
        Path to the directory containing cropped cell images.
    """

    # Gather crop files
    crop_files = sorted([f for f in os.listdir(crop_dir) if f.endswith(('.png', '.jpg'))])
    print(f"Found {len(crop_files)} cropped cell(s) in {crop_dir}...")

    widths = []
    heights = []

    for crop_file in crop_files:
        crop_path = os.path.join(crop_dir, crop_file)
        img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Could not read {crop_file} — skipping.")
            continue

        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20, color='blue', alpha=0.7)
    plt.title('Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20, color='red', alpha=0.7)
    plt.title('Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()