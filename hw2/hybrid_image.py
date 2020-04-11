import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift 
import matplotlib.pyplot as plt
from matplotlib.image import imread

IMAGE_DIR = 'hw2_data/task1and2_hybrid_pyramid/'

def make_gaussian(row, col, sigma, is_high):
    res = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            val = np.exp(((i - row // 2) ** 2 + (j - col // 2) ** 2) / (-2 * sigma ** 2))
            if is_high:
                val = 1 - val
            res[i][j] = val
    return res

def freq_filter(img, sigma, is_high):
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             img_copy[i][j] *= pow(-1, i+j)

    img_freq = fftshift(fft2(img))
    freq_pass_func = make_gaussian(img.shape[0], img.shape[1], sigma, is_high)
    img_freq_filtered = img_freq * freq_pass_func
    img_filtered = ifft2(ifftshift(img_freq_filtered)).real

#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             img_filtered[i][j] *= pow(-1, i+j)
    return img_filtered



def hybrid(img_high, img_low, sigma_high, sigma_low):
    img_high_pass = freq_filter(img_high, sigma_high, is_high=True)
    img_low_pass = freq_filter(img_low, sigma_low, is_high=False)

    return img_high_pass + img_low_pass

def get_img(img_name):
    img_rgb = imread(IMAGE_DIR + img_name).astype('float')
    img_gray = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114])
    return img_gray


if __name__ == "__main__":
    img_high = get_img('4_einstein.bmp')
    img_low = get_img('4_marilyn.bmp')
    
    img_hybrid = hybrid(img_high, img_low, sigma_high=25, sigma_low=10)
    plt.imshow(img_hybrid, cmap='gray', vmin=0, vmax=255)

#     plt.show()
    plt.draw()
    plt.pause(1)
    input("<hit Enter to close>")
    plt.close(plt.figure())


