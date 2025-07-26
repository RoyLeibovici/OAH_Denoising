import os
import shutil
import numpy as np
import cv2
from cellSAM import cellsam_pipeline


def segment_and_sort_frames(frames_dir, output_dir):
    """
    Processes each frame with cellSAM and logs whether a mask was detected or not.
    Then, sorts them into 'cells', 'background', and 'masks' folders accordingly.

    Parameters:
    -----------
    frames_dir : str
        Path to the directory containing extracted frame PNGs.
    output_dir : str
        Path where "masks", "cells", and "background" folders will be created.
    """

    # Prepare output folders
    masks_dir = os.path.join(output_dir, "masks")
    with_mask_dir = os.path.join(output_dir, "frames with mask")
    no_mask_dir = os.path.join(output_dir, "frames without mask")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(with_mask_dir, exist_ok=True)
    os.makedirs(no_mask_dir, exist_ok=True)

    # Gather frame list
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    print(f"Found {len(frame_files)} frame(s) in {frames_dir}...")

    # Logs
    frames_with_masks = []
    frames_without_masks = []

    masks_by_frame = {}

    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)

        # Read image
        img_gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
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
        frame_path = os.path.join(frames_dir, frame_file)
        shutil.copy(frame_path, os.path.join(with_mask_dir, frame_file))

        mask = masks_by_frame[frame_file]
        mask_filename = frame_file.replace(".png", ".npy")
        np.save(os.path.join(masks_dir, mask_filename), mask)

    for frame_file in frames_without_masks:
        frame_path = os.path.join(frames_dir, frame_file)
        shutil.copy(frame_path, os.path.join(no_mask_dir, frame_file))

    print(f"Done! {len(frames_with_masks)} with masks, {len(frames_without_masks)} without masks.")

    # Write Log
    log_path = os.path.join(output_dir, "segmentation_log.txt")

    with open(log_path, "w") as f:
        f.write(f"{'WITH_MASK':<25} | {'NO_MASK':<25}\n")
        f.write(f"{'-'*25}-+-{'-'*25}\n")

        max_len = max(len(frames_with_masks), len(frames_without_masks))
        for i in range(max_len):
            mask_frame = frames_with_masks[i] if i < len(frames_with_masks) else ""
            no_mask_frame = frames_without_masks[i] if i < len(frames_without_masks) else ""
            f.write(f"{mask_frame:<25} | {no_mask_frame:<25}\n")

def segment_and_sort_frames_batched(frames_dir, output_dir, batch_size=100):
    """
    Segments frames in batches, saves after each batch, and maintains a runtime log
    to allow resuming from the last checkpoint.
    """

    # Output dirs
    masks_dir = os.path.join(output_dir, "masks")
    with_mask_dir = os.path.join(output_dir, "frames with mask")
    no_mask_dir = os.path.join(output_dir, "frames without mask")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(with_mask_dir, exist_ok=True)
    os.makedirs(no_mask_dir, exist_ok=True)

    # Runtime log path
    runtime_log_path = os.path.join(output_dir, "runtime_log.txt")

    # Load already processed frames
    processed_frames = set()
    if os.path.exists(runtime_log_path):
        with open(runtime_log_path, "r") as f:
            processed_frames = set(line.strip() for line in f if line.strip())

    # Gather all frames
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

    # Filter out already processed ones
    frame_files = [f for f in frame_files if f not in processed_frames]
    total_to_process = len(frame_files)
    print(f"Found {total_to_process} unprocessed frame(s) in {frames_dir}...")

    # Process in batches
    for start_idx in range(0, total_to_process, batch_size):
        batch = frame_files[start_idx:start_idx + batch_size]
        print(f"\nProcessing batch {start_idx}–{start_idx + len(batch) - 1}...")

        frames_with_masks = []
        frames_without_masks = []
        masks_by_frame = {}

        for frame_file in batch:
            frame_path = os.path.join(frames_dir, frame_file)

            img_gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"[WARNING] Could not read {frame_file} — skipping.")
                continue

            img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

            try:
                mask = cellsam_pipeline(
                    img_rgb,
                    use_wsi=False,
                    low_contrast_enhancement=False,
                    gauge_cell_size=False,
                )

                if isinstance(mask, np.ndarray):
                    frames_with_masks.append(frame_file)
                    masks_by_frame[frame_file] = mask
                else:
                    frames_without_masks.append(frame_file)

            except Exception as e:
                print(f"[ERROR] Failed processing {frame_file}: {e}")
                frames_without_masks.append(frame_file)

        # Save batch results
        for frame_file in frames_with_masks:
            shutil.copy(os.path.join(frames_dir, frame_file), os.path.join(with_mask_dir, frame_file))
            np.save(os.path.join(masks_dir, frame_file.replace(".png", ".npy")), masks_by_frame[frame_file])

        for frame_file in frames_without_masks:
            shutil.copy(os.path.join(frames_dir, frame_file), os.path.join(no_mask_dir, frame_file))

        # Update runtime log
        with open(runtime_log_path, "a") as f:
            for frame_file in (frames_with_masks + frames_without_masks):
                f.write(frame_file + "\n")

        print(f"Batch {start_idx}–{start_idx + len(batch) - 1} done: "
              f"{len(frames_with_masks)} with masks, {len(frames_without_masks)} without.")

    print("\nAll batches processed and saved!")
