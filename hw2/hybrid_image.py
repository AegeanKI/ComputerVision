import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift 
import matplotlib.pyplot as plt
from matplotlib.image import imread

IMAGE_DIR = 'hw2_data/task1and2_hybrid_pyramid/'

def make_gaussian(row, col, sigma, is_high):
    res = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            val = np.exp(-1 * ((i - row // 2) ** 2 + (j - col // 2) ** 2) / (2 * (sigma ** 2)))
            if is_high:
                val = 1 - val
            res[i][j] = val
    return res

def move_center(img):
    row = img.shape[0]
    col = img.shape[1]
    img_copy = np.array(img)
    for i in range(row):
        for j in range(col):
            img_copy[i][j] *= pow(-1, i+j)

    return img_copy
    

def filt(img, freq_pass_func):
    img_freq = fftshift(fft2(img))
#     img_freq = fft2(img)
#     freq_pass_func = fftshift(fft2(freq_pass_func))
    img_freq_filted = img_freq * freq_pass_func
    img_filted = ifft2(ifftshift(img_freq_filted)).real
#     img_filted = ifft2(img_freq_filted).real
    return img_filted

def freq_filter(img, sigma, is_high):
    row = img.shape[0]
    col = img.shape[1]
    img_rgb = [i.reshape(row, col) for i in np.dsplit(img, 3)]
    img_rgb = [move_center(i) for i in img_rgb]

    freq_pass_func = make_gaussian(row, col, sigma, is_high)
    img_filted_rgb = [filt(i, freq_pass_func) for i in img_rgb]

    img_filted_rgb = [move_center(i) for i in img_filted_rgb]
    img_filted = np.stack(img_filted_rgb, axis=2)
    return img_filted


def hybrid(img_high, img_low, sigma_high, sigma_low):
    img_high_pass = freq_filter(img_high, sigma_high, is_high=True)
#     img_low_pass = freq_filter(img_low, sigma_low, is_high=False)

    return img_high_pass + img_low_pass

def get_img(img_name):
    img_rgb = imread(IMAGE_DIR + img_name).astype('int')
    return img_rgb


if __name__ == "__main__":
    img_low = get_img('4_einstein.bmp')
    img_high = get_img('4_marilyn.bmp')
    
    img_hybrid = hybrid(img_high, img_low, sigma_high=2, sigma_low=1)
    plt.imshow(img_hybrid)

#     plt.show()
    plt.draw()
    plt.pause(1)
    input("<hit Enter to close>")
    plt.close(plt.figure())


