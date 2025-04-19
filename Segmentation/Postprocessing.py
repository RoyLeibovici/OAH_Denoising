import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def process_frames(frames_dir, masks_dir, output_dir, patch_size=64, mask_threshold_percentage=5):
    """
    Processes frame and mask images to extract cell regions and background patches.

    Args:
        frames_dir (str): Directory containing the frame images.
        masks_dir (str): Directory containing the mask images.
        output_dir (str): Directory to save the 'cells' and 'background' subimages.
        patch_size (int): Size (n) of the square subimages (n x n).
        mask_threshold_percentage (int): Percentage of mask coverage allowed in a background patch.
    """
    cells_output_dir = os.path.join(output_dir, 'cells')
    background_output_dir = os.path.join(output_dir, 'background')
    os.makedirs(cells_output_dir, exist_ok=True)
    os.makedirs(background_output_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(frames_dir) if not f.startswith('mask_')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.startswith('mask_')])

    frame_nums = {f.split('.')[0].replace('frame_', ''): os.path.join(frames_dir, f) for f in frame_files}
    mask_nums = {f.split('.')[0].replace('mask_frame_', ''): os.path.join(masks_dir, f) for f in mask_files}

    for frame_num, frame_path in frame_nums.items():
        if frame_num in mask_nums:
            mask_path = mask_nums[frame_num]
            frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            if frame is None or mask is None:
                print(f"Error loading frame or mask for frame number: {frame_num}")
                continue

            frame_height, frame_width, _ = frame.shape
            mask_height, mask_width, _ = mask.shape

            if frame_height != mask_height or frame_width != mask_width:
                print(f"Frame and mask dimensions do not match for frame number: {frame_num}")
                continue

            # Extract cells using the mask
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
            cell_mask = mask_gray  # Binary mask: 1 where cell is present, 0 elsewhere
            cells = frame[cell_mask]  # Extract the pixels of the cells.

            if cells.size > 0: #check if any cells were found
                cells_filename = f"cells_{frame_num}.png"
                # Reshape 'cells' to a proper image format before saving
                # Find the bounding box of the cell mask to crop a tight image.
                y_coords, x_coords = np.where(cell_mask)
                if y_coords.size > 0 and x_coords.size > 0:  # Check to prevent errors if no cells are found.
                    min_y, max_y = np.min(y_coords), np.max(y_coords)
                    min_x, max_x = np.min(x_coords), np.max(x_coords)
                    cropped_cell_img = frame[min_y:max_y+1, min_x:max_x+1] # +1 to include the last row and column
                    cv2.imwrite(os.path.join(cells_output_dir, cells_filename), cropped_cell_img)
                else:
                    print(f"No cells found in frame: {frame_num}")

            # Extract background patches using a sliding window
            for y in range(0, frame_height - patch_size + 1, patch_size):
                for x in range(0, frame_width - patch_size + 1, patch_size):
                    sub_mask = mask_gray[y:y + patch_size, x:x + patch_size]
                    mask_area = np.sum(sub_mask > 0)  # Count non-zero pixels in the sub_mask
                    patch_area = patch_size * patch_size
                    mask_percentage = (mask_area / patch_area) * 100

                    if mask_percentage <= mask_threshold_percentage:
                        # Save the n*n patch as background
                        sub_frame = frame[y:y + patch_size, x:x + patch_size]
                        background_filename = f"background_{frame_num}_x{x}_y{y}.png"
                        cv2.imwrite(os.path.join(background_output_dir, background_filename), sub_frame)
        else:
            print(f"No mask found for frame: {frame_num}")

    print("Processing complete.")

if __name__ == "__main__":
    frames_dir = r'G:\My Drive\Shlomi and Roy\Final Project\videos\HTB5-170122_frames'
    masks_dir = r'G:\My Drive\Shlomi and Roy\Final Project\videos\HTB5-170122_frames\masks'
    output_dir = r'G:\My Drive\Shlomi and Roy\Final Project\videos\HTB5-170122_frames'
    patch_size = 64
    mask_threshold_percentage = 5  # Adjust as needed

    process_frames(frames_dir, masks_dir, output_dir, patch_size, mask_threshold_percentage)
