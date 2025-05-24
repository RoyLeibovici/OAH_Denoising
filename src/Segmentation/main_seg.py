from sample_video import sample_frames
from masks import segment_and_sort_frames
from extract_cells import crop_cells
from pathlib import Path


# Define project Paths
current_file = Path(__file__).resolve()
project_path = current_file.parents[2]
Data_path = Path(project_path, "Data")
Videos_path = Path(Data_path, "Input_Videos")
Output_files = Path(Data_path, "Output_files")

for video_path in Videos_path.glob("*.AVI"):
    video_outputs, frames_path = sample_frames(video_path, Output_files, sampling_rate=1)
    # segment_and_sort_frames(frames_path, video_outputs)

    frames_with_masks_path = Path(video_outputs, "frames with mask")
    masks_path = Path(video_outputs, "masks")
    crop_cells(frames_with_masks_path, masks_path, video_outputs)