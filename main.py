
import os
import pandas as pd
import numpy as np
# import skimage.io as io
from openslide import OpenSlide
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
print('libraries imported')
from ImageLoadPreprocessFunctions import *






# ------------------------------ Demonstration ------------------------------


        

image_path_H = 'trial_images/028989_0.tif' # very huge image examble (85.000 x 35.000 x 4)
image_path_M = 'trial_images/008e5c_0.tif' # medium large image examble (around 20.000 x 10.000 x 4)

image = Slide_Open_Resize(image_path_M)
plt.imshow(image); plt.show()
image_cut_off, blank_map = Slide_Cut_off_Resize(image_path_M)
plt.imshow(image_cut_off); plt.show()



image = Slide_Open_Resize(image_path_H)
plt.imshow(image); plt.show()
image_cut_off, blank_map = Slide_Cut_off_Resize(image_path_H)
plt.imshow(image_cut_off); plt.show()


image_tiles = Optimized_Slide_Pack(image_path_M)
for im in image_tiles:
    im = np.array(im)
    plt.imshow(im); plt.show()