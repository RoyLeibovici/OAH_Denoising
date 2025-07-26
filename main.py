import argparse
import sys
import os


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

    # Create output directory if it doesn't exist
    try:
        os.makedirs(args.output, exist_ok=True)
    except OSError as e:
        print(f"Error: Cannot create output directory '{args.output}': {e}", file=sys.stderr)
        sys.exit(1)

    # Check if output directory is writable
    if not os.access(args.output, os.W_OK):
        print(f"Error: Output directory '{args.output}' is not writable.", file=sys.stderr)
        sys.exit(1)


def main():
    # Main function that handles argument parsing
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_arguments(args)

    # Print parsed arguments (remove this in production)
    print(f"Input path: {args.input}")
    print(f"Output/Working directory: {args.output}")
    print(f"Mode: {args.mode}")

    # Your main application logic goes here
    if args.mode == 'train':
        print("Running in training mode...")
        # Add your training logic
    elif args.mode == 'cells':
        print("Running in cells mode...")
        # Add your cells processing logic
    elif args.mode == 'frames':
        print("Running in frames mode...")
        # Add your frames processing logic
    elif args.mode == 'video':
        print("Running in video mode...")
        # Add your video processing logic


if __name__ == "__main__":
    main()