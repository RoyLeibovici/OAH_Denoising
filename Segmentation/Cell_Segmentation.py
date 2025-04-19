import numpy as np
from cellSAM import segment_cellular_image
import matplotlib.pyplot as plt
from PIL import Image


path = r'G:\My Drive\Shlomi and Roy\Final Project\code\frames\frame_0328.png'
path2 = r'C:\Users\Roy Leibovici\PycharmProjects\OAH_Denoising\cellSAM\sample_imgs\Yeaz.png'
img = np.array(Image.open(path))
#img = np.load(path2, allow_pickle=True)

mask, embedding, bounding_boxes = segment_cellular_image(img, device='cpu')
plt.imsave('mask.png', mask)

