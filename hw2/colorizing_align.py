import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from matplotlib.image import imread
import gc

IMAGE_DIR = 'hw2_data/task3_colorizing/'
IMAGE = [
    'cathedral.jpg', 'emir.tif', 'icon.tif', 'lady.tif', 'melons.tif',
    'monastery.jpg', 'nativity.jpg', 'onion_church.tif',
    'three_generations.tif', 'tobolsk.jpg', 'train.tif', 'village.tif',
    'workshop.tif'
]


def get_img(img_name):
    img_rgb = imread(IMAGE_DIR + img_name).astype('int')
    return img_rgb


def ncc(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return np.sum(((a / np.linalg.norm(a)) * (b / np.linalg.norm(b))))


def nccAlign(a, b, t):
    min_ncc = -1
    ivalue = np.linspace(-t, t, 2 * t, dtype=int)
    jvalue = np.linspace(-t, t, 2 * t, dtype=int)
    for i in ivalue:
        for j in jvalue:
            nccDiff = ncc(a, np.roll(b, [i, j], axis=(0, 1)))
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i, j]
    return output


def coloring(img, imname):
    # imname = '01047u.tif'
    # img = tifffile.imread(imname)
    img = np.asarray(img)

    # plt.imshow(img)
    '''
    w, h = img.shape
    img = img[int(w * 0.01):int(w - w * 0.02), int(h * 0.05):int(h - h * 0.05)]
    w, h = img.shape
    height = int(w / 3)
    blue_ = img[0:height, :]
    green_ = img[height:2 * height, :]
    red_ = img[2 * height:3 * height, :]
    print(img.size)
    if imname[-1] == 'f':
        img = imresize(img, 10)
    img = np.asarray(img, dtype='uint8')
    print(img.shape)
    plt.imshow(img)
    '''
    w, h = img.shape
    height = int(w / 3)
    blue = img[0:height, :]
    green = img[height:2 * height, :]
    red = img[2 * height:3 * height, :]
    # plt.figure()
    # plt.imshow(blue)
    # plt.figure()
    # plt.imshow(green)
    # plt.figure()
    # plt.imshow(red)

    alignGtoB = nccAlign(blue, green, 20)
    alignRtoB = nccAlign(blue, red, 20)
    print(alignGtoB, alignRtoB)
    # g = np.roll(green, [alignGtoB[0] * 10, alignGtoB[1] * 10], axis=(0, 1))
    # r = np.roll(red, [alignRtoB[0] * 10, alignRtoB[1] * 10], axis=(0, 1))
    g = np.roll(green, alignGtoB, axis=(0, 1))
    r = np.roll(red, alignRtoB, axis=(0, 1))
    coloured = (np.dstack((r, g, blue)))
    coloured = coloured[int(coloured.shape[0] * 0.05):int(
        coloured.shape[0] - coloured.shape[0] * 0.05),
                        int(coloured.shape[1] * 0.05):int(
                            coloured.shape[1] - coloured.shape[1] * 0.05)]
    # tifffile.imsave('001047u.jpg', coloured)
    return coloured


if __name__ == "__main__":
    '''
    Low resolution: i = 0, 5, 6, 9
    '''
    low_resolution_idx = [0, 5, 6, 9]
    # i = 9
    for i in low_resolution_idx:
        img = get_img(IMAGE[i])
        # img = get_img(inputimg)
        # if np.amax(img) > 255:
        #     img = img // 2**8
        # print(img)
        # plt.imshow(img)
        # plt.show()

        # coloring(img, inputimg)
        img_colored = coloring(img, IMAGE[i])
        plt.imshow(img_colored)
        # save_path = 'coloring_align/{}.jpg'.format(i)
        # plt.savefig(save_path)
        # plt.imshow(img_colored)

        plt.show()
    # garbage collecter
    # gc.collect()