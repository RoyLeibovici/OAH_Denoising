import argparse
import sys
import os
from pathlib import Path
from src.Denoising.autoencoder import ConvAutoencoder_gated
from src.Denoising.eval_denoising import evaluate_autoencoder
from src.Denoising.data_preperation import resize_cells, discard_cells
from src.Segmentation.masks import segment_and_sort_frames_batched
from src.Segmentation.extract_cells import crop_cells
from src.Segmentation.sample_video import sample_frames


def create_parser():
    # Create and configure the argument parser.
    parser = argparse.ArgumentParser(
        description='Model Activation Parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input /path/to/frames --workdir ./results --mode frames
  python main.py -i ./cell_images -w ./output -m cells
  python main.py --input data/videos --workdir results --mode video
  python main.py -i ./small_cells -w ./work -m cells-resize

Input requirements by mode:
  video       - Directory containing .avi video files
  frames      - Directory containing .png frame images  
  cells       - Directory containing .png cell images (128x128)
  cells-resize- Directory containing .png cell images (any size, will be resized)
  train       - Not implemented yet
        """
    )

    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input images or video directory'
    )

    parser.add_argument(
        '--workdir', '-w',
        type=str,
        required=True,
        help='Working directory for processing and output files'
    )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        required=True,
        choices=['video', 'frames', 'cells', 'cells-resize', 'train'],
        help='Operation mode: cells, cells-resize,frames, video or train'
    )

    return parser


def validate_arguments(args):
    # Validate the parsed arguments and exit if invalid.

    # Check if input path exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Check if input is a directory or file (depending on your needs)
    if not (os.path.isdir(args.input) or os.path.isfile(args.input)):
        print(f"Error: Input path '{args.input}' is neither a file nor directory.", file=sys.stderr)
        sys.exit(1)

    # Create work directory if it doesn't exist
    try:
        os.makedirs(args.workdir, exist_ok=True)
    except OSError as e:
        print(f"Error: Cannot create work directory '{args.workdir}': {e}", file=sys.stderr)
        sys.exit(1)

    # Check if work directory is writable
    if not os.access(args.workdir, os.W_OK):
        print(f"Error: Work directory '{args.workdir}' is not writable.", file=sys.stderr)
        sys.exit(1)


def main():
    # Main function that handles argument parsing
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_arguments(args)

    print(f"Input path: {args.input}")
    print(f"Output/Working directory: {args.workdir}")
    print(f"Mode: {args.mode}")
    weights_path = Path("Denoising") / "weights" / "denoiser.weights.pth"

    if args.mode == 'train':
        print("Running in training mode...")
        # Add train logic
    elif args.mode == 'cells':
        print("Running in cells mode...")
        evaluate_autoencoder(ConvAutoencoder_gated, weights_path, args.input, args.workdir)

    elif args.mode == 'cells-resize':
        print("Running in cells-resize mode...")
        resized_cells_path = Path(args.workdir) / 'resized input'
        resize_cells(args.input, resized_cells_path)
        evaluate_autoencoder(ConvAutoencoder_gated, weights_path, resized_cells_path, args.workdir)

    elif args.mode == 'frames':
        print("Running in frames mode...")
        segment_and_sort_frames_batched(args.input, args.workdir)

        frames_with_masks_path = Path(args.workdir, "frames with mask")
        frames_with_masks_path.mkdir(parents=True, exist_ok=True)
        masks_path = Path(args.workdir, "masks")
        masks_path.mkdir(parents=True, exist_ok=True)

        crop_cells(frames_with_masks_path, masks_path, args.workdir)
        cropped_cells_path = Path(args.workdir) / 'original cells'
        filtered_cells_path = Path(args.workdir) / 'filtered cells'
        discard_cells(cropped_cells_path, filtered_cells_path, min_dim=30)

        resized_cells_path = Path(args.workdir) / 'resized input'
        resize_cells(filtered_cells_path, resized_cells_path)

        evaluate_autoencoder(ConvAutoencoder_gated, weights_path, resized_cells_path, args.workdir)

    elif args.mode == 'video':
        print("Running in video mode...")
        _, frames_path = sample_frames(args.input, args.workdir)
        segment_and_sort_frames_batched(frames_path, args.workdir)

        frames_with_masks_path = Path(args.workdir, "frames with mask")
        frames_with_masks_path.mkdir(parents=True, exist_ok=True)
        masks_path = Path(args.workdir, "masks")
        masks_path.mkdir(parents=True, exist_ok=True)

        crop_cells(frames_with_masks_path, masks_path, args.workdir)
        cropped_cells_path = Path(args.workdir) / 'original cells'
        filtered_cells_path = Path(args.workdir) / 'filtered cells'
        discard_cells(cropped_cells_path, filtered_cells_path, min_dim=30)

        resized_cells_path = Path(args.workdir) / 'resized input'
        resize_cells(filtered_cells_path, resized_cells_path)

        evaluate_autoencoder(ConvAutoencoder_gated, weights_path, resized_cells_path, args.workdir)



if __name__ == "__main__":
    main()