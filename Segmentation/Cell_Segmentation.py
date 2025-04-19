import numpy as np
from cellSAM import segment_cellular_image

path = r'G:\My Drive\Shlomi and Roy\Final Project\code\frames\frame_0004.png'
img = np.load(path)
mask, _, _ = segment_cellular_image(img, device='cuda')