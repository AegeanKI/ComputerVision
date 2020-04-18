import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift 
import matplotlib.pyplot as plt
from matplotlib.image import imread

IMAGE_DIR = 'hw2_data/task1and2_hybrid_pyramid/'
IMAGE_HIGH = ['0_Afghan_girl_after.jpg', '1_bicycle.bmp', '2_bird.bmp', '3_cat.bmp',
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


def hybrid(img_high, img_low, sigma_high, sigma_low, gaussian):
    img_high_pass = freq_filter(img_high, sigma_high, gaussian, is_high=True)
    img_low_pass = freq_filter(img_low, sigma_low, gaussian, is_high=False)

    return img_high_pass + img_low_pas


if __name__ == "__main__":
    i = 5
    img_high = get_img(IMAGE_HIGH[i])
    img_low = get_img(IMAGE_LOW[i])
    
    # img_hybrid = hybrid(img_high, img_low, sigma_high=24, sigma_low=10, gaussian=True)
    img_hybrid = hybrid(img_high, img_low, sigma_high=16, sigma_low=24, gaussian=False)
    plt_valid_show(img_hybrid)

    # plt.draw()
    # plt.pause(1)
    # input("<hit Enter to close>")
    # plt.close(plt.figure())


