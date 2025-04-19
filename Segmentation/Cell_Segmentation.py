import numpy as np
from cellSAM import segment_cellular_image, cellsam_pipeline

import matplotlib.pyplot as plt
from PIL import Image


path = r'G:\My Drive\Shlomi and Roy\Final Project\videos\HTB5-170122_frames\frame_000004.png'
path2 = r'C:\Users\Roy Leibovici\PycharmProjects\OAH_Denoising\cellSAM\sample_imgs\Yeaz.png'

img = np.array(Image.open(path))
print(np.shape(img))

"""mask, embedding, bounding_boxes = segment_cellular_image(img, device='cpu', bbox_threshold=0.2)
"""

# Run inference pipeline
mask = cellsam_pipeline(
    img, use_wsi=False, low_contrast_enhancement=False, gauge_cell_size=False, bbox_threshold=0.1)

# Visualize results
plt.imsave('mask.png', mask)
