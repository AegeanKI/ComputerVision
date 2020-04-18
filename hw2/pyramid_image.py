import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift 
import matplotlib.pyplot as plt
from matplotlib.image import imread

IMAGE_DIR = 'hw2_data/task1and2_hybrid_pyramid/'
IMAGE_HIGH = ['0_Afghan_girl_after.jpg', '1_bicycle.bmp','2_bird.bmp', '3_cat.bmp',
              '4_einstein.bmp', '5_fish.bmp', '6_makeup_after.jpg']
IMAGE_LOW = ['0_Afghan_girl_before.jpg', '1_motorcycle.bmp', '2_plane.bmp', '3_dog.bmp',
             '4_marilyn.bmp', '5_submarine.bmp', '6_makeup_before.jpg']

def get_img(img_name):
    img_rgb = imread(IMAGE_DIR + img_name).astype('int')
    return img_rgb


def plt_valid_show(img):
    img_copy = np.array(img)
    img_copy[img_copy < 0] = 0
    img_copy[img_copy > 255] = 255
    plt.imshow(img_copy)
    plt.show()


def make_filter(row, col, sigma, is_high, gaussian):
    res = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            distance = (i - row // 2) ** 2 + (j - col // 2) ** 2
            if gaussian:
                val = np.exp(-1 * distance / (2 * (sigma ** 2)))
            else:
                val = 1 if distance > sigma ** 2 else 0
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

        plt_valid_show(smooth_img)
        plt_valid_show(subsample)
        plt_valid_show(laplacian_img)


if __name__ == "__main__":
    i = 5
    img_high = get_img(IMAGE_HIGH[i])
    img_low = get_img(IMAGE_LOW[i])

    img_pyramid = pyramid(img_low, 3)
    # plt.imshow(img_pyramid)

#     plt.show()
    # plt.draw()
    # plt.pause(1)
    # input("<hit Enter to close>")
    # plt.close(plt.figure())
