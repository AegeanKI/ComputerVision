import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift 
import matplotlib.pyplot as plt
from matplotlib.image import imread
import gc

IMAGE_DIR = 'hw2_data/task3_colorizing/'
IMAGE= [ 'cathedral.jpg' ,'emir.tif', 'icon.tif', 'lady.tif', 'melons.tif', 'monastery.jpg', 'nativity.jpg', 'onion_church.tif', 'three_generations.tif', 'tobolsk.jpg', 'train.tif', 'village.tif', 'workshop.tif']

def get_img(img_name):
    img_rgb = imread(IMAGE_DIR + img_name).astype('int')
    return img_rgb

def coloring(img):
    print(img.shape)
    row = img.shape[0]
    col = img.shape[1]
    split = row//3
    img_b = img[0:split].reshape(split,col,1)
    img_g = img[split:split*2].reshape(split,col,1)
    img_r = img[split*2:split*3].reshape(split,col,1)
    # plt.imshow(img_r)
    # plt.show()
    # plt.imshow(img_g)
    # plt.show()
    # plt.imshow(img_b)
    # plt.show()
    print(img_r.shape)
    img = np.concatenate((img_r,img_g,img_b),axis=2)
    print(img.shape)

    return img

if __name__ == "__main__":
    # i = 8
    for inputimg in IMAGE:
        # img = get_img(IMAGE[i])
        img = get_img(inputimg)
        if np.amax(img) > 255:
            img = img // 2**8 
        print(img)
        plt.imshow(img)
        plt.show()
        
        img_colored = coloring(img)
        plt.imshow(img_colored)

        plt.show()
        # garbage collecter
        gc.collect()
    # plt.draw()
    # plt.pause(1)
    # input("<hit Enter to close>")
    # plt.close(plt.figure())
