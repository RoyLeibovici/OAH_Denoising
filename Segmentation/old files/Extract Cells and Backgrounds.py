import os
import numpy as np
import cv2  # OpenCV for image processing


# Paths
frames_path = r'G:\My Drive\Shlomi and Roy\Final Project\videos\shlomi\SegSortByMedian\cells'
masks_path = r'G:\My Drive\Shlomi and Roy\Final Project\videos\shlomi\SegSortByMedian\masks'
cells_path = r'G:\My Drive\Shlomi and Roy\Final Project\videos\shlomi\SegSortByMedian\cells_only'

# Ensure output directory exists
os.makedirs(cells_path, exist_ok=True)

# Load and sort frame and mask filenames
frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(".png")])
mask_files = sorted([f for f in os.listdir(masks_path) if f.endswith(".npy")])

# Process each frame and its corresponding mask
i = 0
for frame_file, mask_file in zip(frame_files, mask_files):
    print(f"Processing frame {i} out of {len(frame_files)}")
    # Load frame image
    frame_path = os.path.join(frames_path, frame_file)
    frame = cv2.imread(frame_path)

    # Load mask
    mask_path = os.path.join(masks_path, mask_file)
    mask = np.load(mask_path)

    # Loop over each unique cell ID (excluding background 0)
    for cell_id in np.unique(mask):
        if cell_id == 0:
            continue  # skip background

        # Create binary mask for this cell
        cell_mask = (mask == cell_id).astype(np.uint8) * 255

        # Find bounding box of the cell
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            cropped_cell = cv2.bitwise_and(frame, frame, mask=cell_mask)
            cell_roi = cropped_cell[y:y+h, x:x+w]

            # Save the cropped cell
            out_filename = f"{os.path.splitext(frame_file)[0]}_cell{cell_id}.png"
            out_path = os.path.join(cells_path, out_filename)
            cv2.imwrite(out_path, cell_roi)



# Parameters
n = 64  # sub-image size
k = 32  # stride
threshold = 0.05  # max fraction of non-background pixels

# Paths
frames_path = r'G:\My Drive\Shlomi and Roy\Final Project\videos\shlomi\SegSortByOriginal\cells'
masks_path = r'G:\My Drive\Shlomi and Roy\Final Project\videos\shlomi\SegSortByOriginal\masks'
bg_path = r'G:\My Drive\Shlomi and Roy\Final Project\videos\shlomi\SegSortByOriginal\bg'

os.makedirs(bg_path, exist_ok=True)

frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(".png")])
mask_files = sorted([f for f in os.listdir(masks_path) if f.endswith(".npy")])

for frame_file, mask_file in zip(frame_files, mask_files):
    frame = cv2.imread(os.path.join(frames_path, frame_file))
    mask = np.load(os.path.join(masks_path, mask_file))

    height, width = mask.shape
    count = 0

    for y in range(0, height - n + 1, k):
        for x in range(0, width - n + 1, k):
            mask_patch = mask[y:y+n, x:x+n]
            cell_fraction = np.count_nonzero(mask_patch) / (n * n)

            if cell_fraction < threshold:
                frame_patch = frame[y:y+n, x:x+n]
                patch_filename = f"{os.path.splitext(frame_file)[0]}_bg_{count}.png"
                cv2.imwrite(os.path.join(bg_path, patch_filename), frame_patch)
                count += 1
