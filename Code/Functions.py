import os
import pandas as pd
import numpy as np
# import skimage.io as io
from openslide import OpenSlide
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image
import gc
import torch
Image.MAX_IMAGE_PIXELS = None
print('libraries imported')



def report_gpu(): 
    print(torch.cuda.list_gpu_processes()) 
    gc.collect() 
    torch.cuda.empty_cache()



def Slide_Open_Resize(path_image):
    
    try:
        image_resized = np.array(cv2.imread(path_image))
        print ('common')
  
    except:
        print("Image too big!!!, slide approach used instead") 
        im_data = OpenSlide(path_image)

        W_init, H_init = im_data.dimensions
        if H_init > 50000 or W_init > 50000:
            ratio = 10 # desired resizing scale e.g.: resized_image = init_image/ratio
                        # this is the pre-init resizing in order to store and handle the image,
                        # at second step, e.g. when need to feed a CNN model, then we can resize again
                        # in the desired CNN input dimensions
            tiles_per_side = 40
            print('ratio: ', ratio)
        elif  H_init > 25000 or W_init > 25000:
            ratio = 5
            tiles_per_side = 20
            print('ratio: ', ratio)
        elif  H_init > 10000 or W_init > 10000:
            ratio = 2
            tiles_per_side = 10
            print('ratio: ', ratio)
        else:
            ratio = 1
            tiles_per_side = 8
            print('ratio: ', ratio)

        tiles_per_side = 40
        tile_size = int(H_init/tiles_per_side), int(W_init/tiles_per_side)

        image_tiles = []
        for i in range(0,H_init-tile_size[0]+1,tile_size[0]):
                blank_row, row, blank_row_map = [], [], []
                for j in range(0,W_init-tile_size[1]+1,tile_size[1]):
                    tile = np.array(im_data.read_region((j,i),0, (tile_size[1], tile_size[0])))
                    ###plt.imshow(tile); plt.show()
                    tile = cv2.resize(tile, dsize = (int(tile_size[1]/ratio), int(tile_size[0]/ratio)), interpolation=cv2.INTER_CUBIC) # INTER_CUBIC   INTER_NEAREST 
                    row.append(tile)
                image_tiles.append(row)

        image_tiles= np.array(image_tiles)

        image_resized = []
        for i in range(np.shape(image_tiles)[0]):
                row = []
                for j in range(np.shape(image_tiles)[1]):
                        row.append(image_tiles[i,j])
                row = np.concatenate(row, axis = 1)
                image_resized.append(row)
        image_resized = np.concatenate(image_resized, axis = 0)

    return image_resized




def blank_tile_detector (tile, r):
        av, std = np.mean(tile), np.std(tile)
        th = av*r# low: 0.05 - high: 0.35  # a high r can detect mainly the dense object areas, while a low r is mainly for tossing out the total blank areas
        if std <= th:                      # and keeping the less dense areas
            return -1 #blank tile detected
        else:
            return 0


# my function:
#             Cut_off_Resize
# Comparing to previous approach, via our blank_tile_detector, it can toss out useless blank areas
# and thus the important areas/objects of the final image will have higher resolution!
# However, the distances and the initial locations between each object of the initial images are lost.
# This can be a slight information loss, if want to use the extracted image into a CNN model. 
# Thus, we extract also an optional "blank_map" for further use, which contain the locations of objects in a map form.

# my function:
#             Cut_off_Resize
# Comparing to previous approach, via our blank_tile_detector, it can toss out useless blank areas
# and thus the important areas/objects of the final image will have higher resolution!
# However, the distances and the initial locations between each object of the initial images are lost.
# This can be a slight information loss, if want to use the extracted image into a CNN model. 
# Thus, we extract also an optional "blank_map" for further use, which contain the locations of objects in a map form.

def Slide_Cut_off_Resize(path_image):
    
    im_data = OpenSlide(path_image)
    
    W_init, H_init = im_data.dimensions
    if H_init > 50000 or W_init > 50000:
        ratio = 10 # desired resizing scale e.g.: resized_image = init_image/ratio
                    # this is the pre-init resizing in order to store and handle the image,
                    # at second step, e.g. when need to feed a CNN model, then we can resize again
                    # in the desired CNN input dimensions
        tiles_per_side = 40
        print('ratio: ', ratio)
    elif  H_init > 25000 or W_init > 25000:
        ratio = 5
        tiles_per_side = 20
        print('ratio: ', ratio)
    elif  H_init > 10000 or W_init > 10000:
        ratio = 2
        tiles_per_side = 10
        print('ratio: ', ratio)
    else:
        ratio = 1
        tiles_per_side = 8
        print('ratio: ', ratio)
    
    tile_size = int(H_init/tiles_per_side), int(W_init/tiles_per_side)

    image_tiles = []
    blank_map = []
    for i in range(0,H_init-tile_size[0]+1,tile_size[0]):
            blank_row, row, blank_row_map = [], [], []
            for j in range(0,W_init-tile_size[1]+1,tile_size[1]):
                tile = np.array(im_data.read_region((j,i),0, (tile_size[1], tile_size[0])))
                
                
                blank_row_map.append(blank_tile_detector (tile, 0.05)) # <<<<<< blank tile detection

                
                tile_r = cv2.resize(tile, dsize = (int(tile_size[1]/ratio), int(tile_size[0]/ratio)), interpolation=cv2.INTER_CUBIC) # INTER_CUBIC   INTER_NEAREST 
                row.append(tile_r)
            
                del tile # free memory
                gc.collect()  
            
            blank_map.append(blank_row_map)
            image_tiles.append(row)

            del row # free memory
            gc.collect() 
            
    blank_map = np.array(blank_map)
    image_tiles= np.array(image_tiles)

    # cut off function
    image_cut_off = []
    for i in range(np.shape(blank_map)[0]):
        if list(blank_map[i]).count(-1) != np.shape(blank_map)[1]:
            row = []
            for j in range(np.shape(blank_map)[1]):
                if list(blank_map[:,j]).count(-1) != np.shape(blank_map)[0]:
                    row.append(image_tiles[i,j])
            row = np.concatenate(row, axis = 1)
            image_cut_off.append(row)
            
            del row # free memory
            gc.collect() 
            
    image_cut_off = np.concatenate(image_cut_off, axis = 0)
    
    del image_tiles # free memory
    gc.collect()
    
    del im_data
    gc.collect()   
    
    report_gpu()
    
    return image_cut_off, blank_map




# my function:
#             Optimized_Slide_Pack
# It can be used for slide packs extraction given an initial huge image.
# Via our hand-crafted blank_tile_detector the final extracted slide pack will be composed by dense object areas.
#     this is significant when need to feed the slides into a CNN model since it will obtain much area information.
#     Comparing to other approaches, with this feed type, a CNN can focus on local object areas, 
#     exploiting much more higher resolutions of the initial huge image.

# It can also be used as a random crop generator (by some minor modifications) for huge initial images, for data augmentation purposes.

def Optimized_Slide_Pack(path_image):

        im_data = OpenSlide(path_image)

        W_init, H_init = im_data.dimensions
        if H_init > 50000 or W_init > 50000:
            ratio = 10 # desired resizing scale e.g.: resized_image = init_image/ratio
                        # this is the pre-init resizing in order to store and handle the image,
                        # at second step, e.g. when need to feed a CNN model, then we can resize again
                        # in the desired CNN input dimensions
            tiles_per_side = 20
            print('ratio: ', ratio)
        elif  H_init > 25000 or W_init > 25000:
            ratio = 5
            tiles_per_side = 10
            print('ratio: ', ratio)
        elif  H_init > 10000 or W_init > 10000:
            ratio = 2
            tiles_per_side = 5
            print('ratio: ', ratio)
        else:
            ratio = 1
            tiles_per_side = 4
            print('ratio: ', ratio)

        Av_S = int((W_init + H_init)/2)
        tile_size = int(Av_S/tiles_per_side), int(Av_S/tiles_per_side)

        image_tiles = []
        for i in range(0,H_init-tile_size[0]+1,tile_size[0]):
                for j in range(0,W_init-tile_size[1]+1,tile_size[1]):
                    tile = np.array(im_data.read_region((j,i),0, (tile_size[1], tile_size[0])))

                    if blank_tile_detector (tile, 0.35) == 0: #<<<<<<<<< dense object detection
                        image_tiles.append(tile)

        image_tiles = np.array(image_tiles)
        return image_tiles
