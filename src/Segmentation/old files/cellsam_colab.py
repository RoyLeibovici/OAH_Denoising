# -*- coding: utf-8 -*-
"""CellSAM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_5UuY_OvzNiIcLB4W6Af_xpzFAz-AYKf
"""

#@title Imports
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from tqdm import tqdm
from google.colab import drive

# Commented out IPython magic to ensure Python compatibility.
#@title Mount Drive
drive.mount('/content/drive/')

# %cd '/content/drive/My Drive/Shlomi and Roy/Final Project/videos/HTB5-170122_frames'

path = os.getcwd()
print('path: ' + path)

#@title Import cellSAM
#!pip install git+https://github.com/vanvalenlab/cellSAM.git
import cellSAM
from cellSAM import cellsam_pipeline
from cellSAM import segment_cellular_image, get_model, get_local_model
from cellSAM import CellSAM

def cellsam_pipeline(
        img,
        chunks=256,
        model_path=None,
        bbox_threshold=0.4,
        low_contrast_enhancement=False,
        swap_channels=False,
        use_wsi=True,
        gauge_cell_size=False,
        block_size=400,
        overlap=56,
        iou_depth=56,
        iou_threshold=0.5,
):
    """Run the cellsam inference pipeline on `img`.

    Cellsam is capable of segmenting a variety of cells (bacteria,
    eukaryotic, etc.) spanning all forms of microscopy (brightfield,
    phase, autofluorescence, electron microscopy) and
    staining (H&E, PAS, etc.) / multiplexed (codex, mibi, etc.)
    modalities.

    Parameters
    ----------
    img : array_like with shape ``(W, H)`` or ``(W, H, C)``, where C is 1 or 3
        The image to be segmented. For multiple-channel images, `img` should
        have the following format:

          - **Stained images (e.g H&E)**: ``(W, H, C)`` where ``C == 3``
            representing color channels in RGB format.
          - **Multiplexed images**: ``(W, H, C)`` where ``C == 3`` and the
            channel ordering is: ``(blank, nuclear, membrane)``. The
            ``membrane`` channel is optional, in which case a nuclear segmentation
            is returned.
    chunks : int
        TODO: should this be an option?
    model_path : str or pathlib.Path, optional
        Path to the model weights. If `None` (the default), the latest released
        cellsam generalist model is used.

        .. note:: Downloading the model requires internet access

    bbox_threshold : float in range [0, 1], default=0.4
        Threshold for the outputs of Cellfinder, only cells with a confidence higher
        than the threshold will be included. This is the main parameter to
        control precision/recall for CellSAM. For very out of distribution images
        use a value lower than 0.4 and vice versa.
    low_contrast_enhancement : bool, default=False
        Whether to enhance low contrast images, like Livecell images as a preprocessing
        step to improve downstream segmentation.
    swap_channels : bool, default=False
        TODO: this should be removed with loading from file
    use_wsi : bool, default=True
        Whether to use tiling to support large images, default is True.
        Generally, tiling is not required when there are fewer than ~3000
        cells in an image.
    gauge_cell_size : bool, default=False
        Wheter to perform one iteration of segmentation initially, and
        use the results to estimate the sizes of cells and then do another
        round of segmentation using tiling parameters with these results.
    block_size : int
        Size of the tiles when `use_wsi` is `True`. In practice, should
        be in the range ``[256, 2048]``, with smaller tile sizes
        preferred for dense (i.e. many cells/FOV) images.
    overlap : int
        Tile overlap region in which label merges are considered. Must
        be smaller than `block_size`. For reliable tiling, value should
        be large enough to encompass `iou_threshold` of the extent of
        a typical object.
    iou_depth : int
        TODO: Detail effects of this parameter: is this/should this be
        distinct from overlap?
    filter_below_min : bool
        TODO: Detail this parameter - is it necessary?

    Returns
    -------
    segmentation_mask : 2D numpy.ndarray of dtype `numpy.uint32`
        A `numpy.ndarray` representing the segmentation mask for `img`.
        The array is 2D with the same dimensions as `img`, with integer
        labels representing pixels corresponding to cell instances.
        Background is denoted by ``0``.

    Examples
    --------
    Using CellSAM to segment a slice from the `~skimage.data.cells3d` dataset.

    >>> import numpy as np
    >>> import skimage
    >>> data = skimage.data.cells3d()
    >>> data.shape
    (60, 2, 256, 256)

    From the `~skimage.data.cells3d` docstring, ``data`` is a 3D multiplexed
    image with dimensions ``(Z, C, X, Y)`` where the ordering of the channel
    dimension ``C`` is ``(membrane, nuclear)``.
    Start by extracting a 2D slice from the 3D volume. The middle slice is
    chosen arbitrarily:

    >>> img = data[30, ...]

    For multiplexed images, CellSAM expects the channel ordering to be
    ``(blank, nuclear, membrane)``:

    >>> seg = np.zeros((*img.shape[1:], 3), dtype=img.dtype)
    >>> seg[..., 1] = img[1, ...]  # nuclear channel
    >>> seg[..., 2] = img[0, ...]  # membrane channel

    Segment the image with `cellsam_pipeline`. Since this is a small image,
    we'll set ``use_wsi=False``. We'll also forgo any pre/post-processing:

    >>> mask = cellsam_pipeline(seg, use_wsi=False)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_path is not None:
        modelpath = model_path
        model = get_local_model(modelpath)
        model.bbox_threshold = bbox_threshold
        model = model.to(device)
    else:
        model = None

        # To prevent creating model for each block
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = get_model(None)
        model = model.to(device)
        model.eval()


    img = img.astype(np.float32)
    img = normalize_image(img)

    if low_contrast_enhancement:
        img = enhance_low_contrast(img)

    inp = da.from_array(img, chunks=chunks)

    if use_wsi:
        if gauge_cell_size:
            labels = use_cellsize_gaging(inp, model, device)
        else:
            labels = segment_wsi(inp, block_size, overlap, iou_depth, iou_threshold, normalize=False, model=model,
                                 device=device, bbox_threshold=bbox_threshold).compute()
    else:
        labels, embedding, bounding_boxes = segment_cellular_image(inp, model=model, normalize=False, device=device)

    return labels, embedding, bounding_boxes

#@title Segmentation Functions
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from tqdm import tqdm

def process_frames_with_cellsam(frames_dir, output_dir=None, device='GPU'):
    """
    Process all frames in a directory with CellSAM and save the segmentation outputs.

    Parameters:
    -----------
    frames_dir : str
        Path to the directory containing frame images
    output_dir : str or None
        Path to the directory where outputs will be saved
        If None, outputs will be saved in subdirectories of frames_dir
    device : str
        Device to use for inference ('GPU' or 'CPU')

    Returns:
    --------
    dict
        Count of processed frames and paths to output directories
    """
    # Set up output directories
    if output_dir is None:
        output_dir = frames_dir

    # Create only the masks directory since cellsam_pipeline only returns masks
    masks_dir = os.path.join(output_dir, 'masks')

    # Create output directory if it doesn't exist
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    # Get all image files from the frames directory
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    frame_files.sort()  # Sort to process in order

    print(f"Found {len(frame_files)} frames to process")

    # Process each frame
    processed_count = 0

    for frame_file in tqdm(frame_files, desc="Processing frames"):
        try:
            # Get the frame path and name
            frame_path = os.path.join(frames_dir, frame_file)
            frame_name = os.path.splitext(frame_file)[0]  # Get filename without extension

            # Load the image
            img = np.array(Image.open(frame_path))

            # Handle image channels
            if img.shape[-1] == 4:
                img = img[:, :, :3]  # Keep only RGB channels

            # Convert to float32 as required by cellsam_pipeline
            img = img.astype(np.float32)

            # Process with CellSAM
            try:
                # Check if image is valid before processing
                if img is None or img.size == 0:
                    print(f"Invalid image {frame_file}: Image is empty")
                    continue

                mask = cellsam_pipeline(img, use_wsi=False, low_contrast_enhancement=False, gauge_cell_size=False, bbox_threshold=0.2)

                # Verify mask is not None before saving
                if mask is None:
                    print(f"cellsam_pipeline returned None for {frame_file}")
                    continue

                # Save mask
                mask_path = os.path.join(masks_dir, f"mask_{frame_name}.png")
                plt.imsave(mask_path, mask, cmap='viridis')  # Use viridis for instance segmentation

                processed_count += 1

            except Exception as e:
                print(f"Error in processing {frame_file}: {str(e)}")
                continue

            # Clear CUDA cache periodically
            if processed_count % 10 == 0 and device.upper() == 'GPU' and torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing frame {frame_file}: {str(e)}")

    print(f"Processing complete. Processed {processed_count} frames.")
    print(f"- Masks saved to: {masks_dir}")

    return {
        "processed_count": processed_count,
        "masks_dir": masks_dir
    }

# For testing a problematic frame individually
def test_single_frame(frame_path, output_dir=None):
    """
    Test processing a single frame with CellSAM

    Parameters:
    -----------
    frame_path : str
        Path to the frame image file
    output_dir : str or None
        Directory to save the output mask
    """
    try:
        print(f"Testing processing of {frame_path}")

        # Create output directory
        if output_dir is None:
            output_dir = os.path.dirname(frame_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load and preprocess the image
        img = np.array(Image.open(frame_path))
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")

        if img.shape[-1] == 4:
            img = img[:, :, :3]
            print(f"Removed alpha channel. New shape: {img.shape}")

        # Convert to float32 as required by cellsam_pipeline
        img = img.astype(np.float32)

        # Process with cellsam_pipeline
        mask = cellsam_pipeline(img, use_wsi=False, low_contrast_enhancement=True, gauge_cell_size=False)
        print(f"Mask shape: {mask.shape if mask is not None else 'None'}")

        if mask is not None:
            # Save mask
            frame_name = os.path.splitext(os.path.basename(frame_path))[0]
            mask_path = os.path.join(output_dir, f"mask_{frame_name}.png")
            plt.imsave(mask_path, mask, cmap='viridis')
            print(f"Successfully saved mask to {mask_path}")
            return True
        else:
            print("cellsam_pipeline returned None")
            return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

#@title Main usage

frames_dir = '/content/drive/MyDrive/HTB5-170122_frames'
process_frames_with_cellsam(path, device='cuda')