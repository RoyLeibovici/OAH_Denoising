import cv2
import os


def extract_frames(video_path, output_parent_dir=None, sampling_rate=1):
    """
    Extract frames from a video and save them as images in a folder named after the video.

    Parameters:
    -----------
    video_path : str
        Path to the input video file
    output_parent_dir : str or None
        Parent directory where the frames folder will be created
        If None, the folder will be created in the same directory as the video
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

    # Create output directory path
    if output_parent_dir is None:
        output_parent_dir = os.path.dirname(video_path)

    output_folder = os.path.join(output_parent_dir, f"{video_name}_frames")

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

            # Convert BGR frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Save the frame
            cv2.imwrite(output_path, gray_frame)
            saved_count += 1

            # Print progress every 100 saved frames
            if saved_count % 100 == 0:
                print(f"Saved {saved_count} frames...")

        frame_number += 1

    # Release the video capture object
    cap.release()

    print(f"Extraction complete. Saved {saved_count} frames to {output_folder}")
    return saved_count, output_folder


# Example usage:
# Mount Google Drive (if using Colab)
# drive.mount('/content/drive')

# Extract frames from a video
# video_path = '/content/drive/MyDrive/your_video.mp4'  # Update this path
# # The output folder will automatically be named '/content/drive/MyDrive/your_video_frames'

# Extract frames from a video
# absolute_path = r'G:\My Drive\Shlomi and Roy\Final Project\videos'
path = 'D:\JetBrains\PycharmProjects\OAH_shlomi\pythonProject1\HTB5-170122.avi'  # Update this path
extract_frames(path, sampling_rate=1)  # Save every frame

