import os
import shutil
import numpy as np
import cv2
from cellSAM import cellsam_pipeline


def segment_median_and_sort_primal(median_frames_dir, primal_frames_dir, output_dir):
    """
    Segments median-subtracted frames using cellSAM, and sorts corresponding primal frames
    into 'cells' and 'background' based on segmentation results. Also saves masks and logs.

    Parameters:
    -----------
    median_frames_dir : str
        Directory containing median-subtracted grayscale .png frames (input to cellSAM).

    primal_frames_dir : str
        Directory containing original/primal .png frames to be sorted based on results.

    output_dir : str
        Root directory where "masks", "cells", "background", and the log file will be saved.
    """

    # Prepare output folders
    masks_dir = os.path.join(output_dir, "masks")
    cells_dir = os.path.join(output_dir, "cells")
    background_dir = os.path.join(output_dir, "background")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(cells_dir, exist_ok=True)
    os.makedirs(background_dir, exist_ok=True)

    # Gather median-subtracted frame list
    frame_files = sorted([f for f in os.listdir(median_frames_dir) if f.endswith(".png")])
    print(f"Found {len(frame_files)} median-subtracted frames in {median_frames_dir}...")

    # Logs
    frames_with_masks = []
    frames_without_masks = []

    masks_by_frame = {}

    for idx, frame_file in enumerate(frame_files):
        median_frame_path = os.path.join(median_frames_dir, frame_file)
        primal_frame_path = os.path.join(primal_frames_dir, frame_file)

        # Read median-subtracted grayscale image
        img_gray = cv2.imread(median_frame_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"[WARNING] Could not read {frame_file} — skipping.")
            continue

        # Convert grayscale to RGB (cellSAM expects 3-channel images)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        try:
            mask = cellsam_pipeline(
                img_rgb,
                use_wsi=False,
                low_contrast_enhancement=False,
                gauge_cell_size=False,
            )

            # Check if the result is a valid NumPy array
            if isinstance(mask, np.ndarray):
                frames_with_masks.append(frame_file)
                masks_by_frame[frame_file] = mask
            else:
                frames_without_masks.append(frame_file)

        except Exception as e:
            print(f"[ERROR] Failed processing {frame_file}: {e}")
            frames_without_masks.append(frame_file)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} frames...")

    print("Segmentation complete! Saving outputs...")

    # Save masks and sort frames into folders
    for frame_file in frames_with_masks:
        primal_path = os.path.join(primal_frames_dir, frame_file)
        shutil.copy(primal_path, os.path.join(cells_dir, frame_file))

        mask = masks_by_frame[frame_file]
        mask_filename = frame_file.replace(".png", ".npy")
        np.save(os.path.join(masks_dir, mask_filename), mask)

    for frame_file in frames_without_masks:
        primal_path = os.path.join(primal_frames_dir, frame_file)
        shutil.copy(primal_path, os.path.join(background_dir, frame_file))

    print(f"Done! {len(frames_with_masks)} with masks, {len(frames_without_masks)} without masks.")

    # Write log
    log_path = os.path.join(output_dir, "segmentation_log.txt")
    with open(log_path, "w") as f:
        f.write(f"{'WITH_MASK':<25} | {'NO_MASK':<25}\n")
        f.write(f"{'-'*25}-+-{'-'*25}\n")
        max_len = max(len(frames_with_masks), len(frames_without_masks))
        for i in range(max_len):
            mask_frame = frames_with_masks[i] if i < len(frames_with_masks) else ""
            no_mask_frame = frames_without_masks[i] if i < len(frames_without_masks) else ""
            f.write(f"{mask_frame:<25} | {no_mask_frame:<25}\n")


# ========== USAGE ==========
# Set your original frames folder, median subtracted frames folder and output root directory manually here
median_frames = r"D:\JetBrains\PycharmProjects\OAH_shlomi\pythonProject1\HTB5-170122_mediansub"
primal_frames = r"D:\JetBrains\PycharmProjects\OAH_shlomi\pythonProject1\HTB5-170122_frames"
output_base = r"D:\JetBrains\PycharmProjects\OAH_shlomi\pythonProject1"

segment_median_and_sort_primal(median_frames, primal_frames, output_base)
