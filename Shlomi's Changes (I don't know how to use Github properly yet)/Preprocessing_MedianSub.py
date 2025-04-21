import cv2
import os
import numpy as np


def extract_frames_with_median_subtraction(video_path, output_parent_dir=None, sampling_rate=1, max_frames_for_median=500):
    """
    Extract frames and subtract the median background frame.

    Parameters:
    -----------
    video_path : str
        Path to the input video file
    output_parent_dir : str or None
        Where to save the frames
    sampling_rate : int
        Save every nth frame
    max_frames_for_median : int
        Max number of frames to use for median background estimation

    Returns:
    --------
    tuple
        (Number of frames saved, Output directory path)
    """
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]

    if output_parent_dir is None:
        output_parent_dir = os.path.dirname(video_path)
    output_folder = os.path.join(output_parent_dir, f"{video_name}_mediansub")
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0, output_folder

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Reading up to {max_frames_for_median} frames for median background...")

    # Step 1: Collect frames for median background
    median_frames = []
    for i in range(min(total_frames, max_frames_for_median)):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        median_frames.append(gray)

    median_stack = np.stack(median_frames, axis=0)
    median_frame = np.median(median_stack, axis=0).astype(np.uint8)
    print(f"Median background computed from {len(median_frames)} frames.")

    # Step 2: Rewind and extract frames with subtraction
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    saved_count = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % sampling_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subtracted = cv2.absdiff(gray, median_frame)
            output_path = os.path.join(output_folder, f"frame_{saved_count:06d}.png")
            cv2.imwrite(output_path, subtracted)
            saved_count += 1

            if saved_count % 100 == 0:
                print(f"Saved {saved_count} median-subtracted frames...")

        frame_number += 1

    cap.release()
    print(f"Done. Saved {saved_count} median-subtracted frames to {output_folder}")
    return saved_count, output_folder


# Example usage:
path = r'D:\JetBrains\PycharmProjects\OAH_shlomi\pythonProject1\HTB5-170122.avi'
extract_frames_with_median_subtraction(path, sampling_rate=1)
