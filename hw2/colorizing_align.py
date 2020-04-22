import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
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


def make_filter(row, col, sigma, is_high, gaussian):
    res = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            distance = (i - row // 2)**2 + (j - col // 2)**2
            if gaussian:
                val = np.exp(-1 * distance / (2 * (sigma**2)))
            else:
                val = 1 if distance <= sigma**2 else 0
            if is_high:
                val = 1 - val
            res[i][j] = val
    return res


def filt(img, freq_pass_func):
    img_freq = fftshift(fft2(img))
    img_freq_filted = img_freq * freq_pass_func
    img_filted = ifft2(ifftshift(img_freq_filted)).real
    return img_filted


def freq_filter(img, sigma, gaussian, is_high):
    row = img.shape[0]
    col = img.shape[1]
    img_rgb = [i.reshape(row, col) for i in np.dsplit(img, 3)]
    freq_pass_func = make_filter(row, col, sigma, is_high, gaussian)
    img_filted_rgb = [filt(i, freq_pass_func) for i in img_rgb]
    img_filted = np.stack(img_filted_rgb, axis=2).astype('int')
    return img_filted


def sampling(img, down):
    return img[::2, ::2] if down else img.repeat(2, axis=0).repeat(2, axis=1)


def sample(img, down):
    row = img.shape[0]
    col = img.shape[1]
    img_rgb = [i.reshape(row, col) for i in np.dsplit(img, 3)]
    img_rgb_sampled = [sampling(i, down) for i in img_rgb]
    img_sampled = np.stack(img_rgb_sampled, axis=2).astype('int')
    return img_sampled


def pyramid(img, layers):
    for i in range(layers):
        smooth_img = freq_filter(img, 24, gaussian=True, is_high=False)
        subsample = sample(smooth_img, down=True)
        upsample = sample(subsample, down=False)
        row = img.shape[0]
        col = img.shape[1]
        laplacian_img = img - upsample[:row, :col]
        img = subsample
    return img


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
            nccDiff = ncc(a[:-i, :-j], np.roll(b, [i, j], axis=(0, 1)))
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


def ssd(a, b):
    return np.sum(np.sum((a - b)**2))


def ssdAlign(a, b, t):
    min_ssd = 1E99
    ivalue = np.linspace(-t, t, 2 * t, dtype=int)
    jvalue = np.linspace(-t, t, 2 * t, dtype=int)
    for i in ivalue:
        for j in jvalue:
            ssdDiff = ssd(a, np.roll(b, [i, j], axis=(0, 1)))
            # print("[",i,",",j,"]")
            if ssdDiff < min_ssd:
                min_ssd = ssdDiff
                output = [i, j]
                # print("min =[",i,",",j,"]")
    return output


def coloring_ssd(img, imname):
    img = np.asarray(img)

    w, h = img.shape
    print(w, h)
    height = int(w / 3)
    blue = img[0:height, :]
    green = img[height:2 * height, :]
    red = img[2 * height:3 * height, :]
    window = h // 5
    print("window=", window)
    blue_w = img[height // 2:height // 2 + window, h // 2:h // 2 + window]
    green_w = img[height // 2 + height:height // 2 + height +
                  window, h // 2:h // 2 + window]
    red_w = img[height // 2 + height * 2:height // 2 + height * 2 +
                window, h // 2:h // 2 + window]
    print(blue_w.shape, green_w.shape, red_w.shape)

    alignGtoB = ssdAlign(blue, green, 40)
    # alignGtoB = ssdAlign(blue_w, green_w, 100)
    alignRtoB = ssdAlign(blue, red, 40)
    # alignRtoB = ssdAlign(blue_w, red_w, 100)
    print(alignGtoB, alignRtoB)
    # g = np.roll(green, [alignGtoB[0] * 10, alignGtoB[1] * 10], axis=(0, 1))
    # r = np.roll(red, [alignRtoB[0] * 10, alignRtoB[1] * 10], axis=(0, 1))
    g = np.roll(green, alignGtoB, axis=(0, 1))
    r = np.roll(red, alignRtoB, axis=(0, 1))
    coloured = (np.dstack((r, g, blue)))
    # coloured = coloured[int(coloured.shape[0] * 0.05):int(
    #     coloured.shape[0] - coloured.shape[0] * 0.05),
    #                     int(coloured.shape[1] * 0.05):int(
    #                         coloured.shape[1] - coloured.shape[1] * 0.05)]
    # tifffile.imsave('001047u.jpg', coloured)
    return coloured


if __name__ == "__main__":
    '''
    Low resolution: i = 0, 5, 6, 9
    '''
    low_resolution_idx = [0, 5, 6, 9]
    high_resolution_idx = [1, 2, 3, 4, 7, 8]

    # for i in low_resolution_idx:
    #     img = get_img(IMAGE[i])
    #     # img = get_img(inputimg)
    #     # if np.amax(img) > 255:
    #     #     img = img // 2**8
    #     # print(img)
    #     # plt.imshow(img)
    #     # plt.show()

    #     # coloring(img, inputimg)
    #     img_colored = coloring(img, IMAGE[i])
    #     plt.imshow(img_colored)
    #     # save_path = 'coloring_align/{}.jpg'.format(i)
    #     # plt.savefig(save_path)
    #     # plt.imshow(img_colored)

    #     plt.show()
    # # garbage collecter
    # # gc.collect()
    for i in low_resolution_idx:
        img = get_img(IMAGE[i])
        if np.amax(img) > 255:
            img = img // 2**8
        # print(img)
        # plt.imshow(img)
        # plt.show()

        # coloring(img, inputimg)
        img_colored = coloring_ssd(img, IMAGE[i])
        plt.imshow(img_colored)
        # save_path = 'coloring_align/{}.jpg'.format(i)
        # plt.savefig(save_path)
        # plt.imshow(img_colored)

        plt.show()
        # garbage collecter
        gc.collect()