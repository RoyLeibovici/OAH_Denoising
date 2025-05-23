import os
import numpy as np
import cv2
import pandas as pd

def crop_cells(primal_frames_dir, masks_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metadata = []

    # Sort frame names for consistency
    frame_files = sorted([
        f for f in os.listdir(primal_frames_dir)
        if f.endswith(('.png', '.jpg'))
    ])
    mask_files = sorted([
        f for f in os.listdir(masks_dir)
        if f.endswith('.npy')
    ])

    for frame_idx, (frame_file, mask_file) in enumerate(zip(frame_files, mask_files)):
        # Load frame and corresponding mask
        frame_path = os.path.join(primal_frames_dir, frame_file)
        mask_path = os.path.join(masks_dir, mask_file)

        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        mask = np.load(mask_path)

        # Get frame name without extension
        frame_name_no_ext = os.path.splitext(frame_file)[0]

        n_cells = mask.max()
        for cell_id in range(1, n_cells + 1):
            cell_mask = (mask == cell_id)
            if not np.any(cell_mask):
                continue

            # Find bounding box
            y_indices, x_indices = np.where(cell_mask)
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            # Crop from the primal frame
            cell_crop = frame[y_min:y_max + 1, x_min:x_max + 1]

            # Generate crop filename
            crop_filename = f"{frame_name_no_ext}_cell_{cell_id:04d}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, cell_crop)

            # Save metadata
            metadata.append({
                "frame_index": frame_idx,
                "frame_name": frame_file,
                "cell_index": cell_id,
                "crop_filename": crop_filename,
                "x_min": int(x_min),
                "y_min": int(y_min),
                "x_max": int(x_max),
                "y_max": int(y_max),
            })

    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

# ========== USAGE ==========

# Set your original frames folder, masks folder and output directory manually here
primal_frames = r"D:\JetBrains\PycharmProjects\OAH_shlomi\pythonProject1\SegSortByMedian\frames with mask"
masks = r"D:\JetBrains\PycharmProjects\OAH_shlomi\pythonProject1\SegSortByMedian\masks"
output_dir = r"D:\JetBrains\PycharmProjects\OAH_shlomi\pythonProject1\crops"

crop_cells(primal_frames, masks, output_dir)
