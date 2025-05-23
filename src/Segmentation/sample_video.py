import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def sample_frames(video_path, output_parent_dir, sampling_rate=1):
    """
    Extract frames from a video and save them as images in a folder named after the video.

    Parameters:
    -----------
    video_path : str
        Path to the input video file
    output_parent_dir : str
        Parent directory where the frames folder will be created
    sampling_rate : int
        Save every nth frame (default: 1, which means save every frame)

    Returns:
    --------
    tuple
        (Number of frames extracted, Path to the output folder)
    """

    # Get video filename without extension
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]


    video_output_folder = os.path.join(output_parent_dir, video_name)
    # Create video output directory if it doesn't exist
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    output_folder = os.path.join(video_output_folder, "frames")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0, output_folder

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"- Filename: {video_filename}")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {frame_count}")
    print(f"- Sampling rate: {sampling_rate} (saving every {sampling_rate} frame)")
    print(f"- Output folder: {output_folder}")

    # Initialize counters
    frame_number = 0
    saved_count = 0

    # Read and process frames
    while True:
        ret, frame = cap.read()

        # Break the loop if we've reached the end of the video
        if not ret:
            break

        # Save frame if it matches our sampling rate
        if frame_number % sampling_rate == 0:
            # Create filename with leading zeros for proper sorting
            output_path = os.path.join(output_folder, f"frame_{saved_count:06d}.png")

            # OpenCV reads in BGR format, convert to RGB for saving
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save the frame
            plt.imsave(output_path, rgb_frame)
            saved_count += 1

            # Print progress every 100 saved frames
            if saved_count % 100 == 0:
                print(f"Saved {saved_count} frames...")

        frame_number += 1

    # Release the video capture object
    cap.release()

    print(f"Extraction complete. Saved {saved_count} frames to {output_folder}")
    return video_output_folder, output_folder
