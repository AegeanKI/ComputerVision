import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift 
import matplotlib.pyplot as plt
from matplotlib.image import imread

IMAGE_DIR = 'hw2_data/task1and2_hybrid_pyramid/'
IMAGE_HIGH = ['0_Afghan_girl_after.jpg','1_bicycle.bmp','2_bird.bmp','3_cat.bmp','4_einstein.bmp','5_fish.bmp','6_makeup_after.jpg']
IMAGE_LOW = ['0_Afghan_girl_before.jpg','1_motorcycle.bmp','2_plane.bmp','3_dog.bmp','4_marilyn.bmp','5_submarine.bmp','6_makeup_before.jpg']


def make_filter(row, col, sigma, is_high, gaussian):
    # row+=2
    # col+=2
    res = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if gaussian:
                val = np.exp(-1 * ((i - row // 2) ** 2 + (j - col // 2) ** 2) / (2 * (sigma ** 2)))
            else:
                if ((i - row // 2) ** 2 + (j - col // 2) ** 2) ** 0.5 > sigma:
                    val = 1
                else:
                    val = 0
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
    # print(img.shape)
    # img = np.concatenate((img,[img[-1]]),axis=0)
    # img = np.concatenate(([img[1]],img),axis=0)
    # print(img.shape)
    # img = np.concatenate((img,img[:,-1].reshape(img.shape[0],1)),axis=1)
    # img = np.concatenate((img[:,1].reshape(img.shape[0],1),img),axis=1)
    # print(img.shape)
    img_freq = fftshift(fft2(img))
    # img_freq = fft2(img)
#     freq_pass_func = fftshift(fft2(freq_pass_func))
    img_freq_filted = img_freq * freq_pass_func
    # plt.imshow(img_freq.astype(int))
    # plt.show()
    # img_filted = ifft2(img_freq_filted).real
    img_filted = ifft2(ifftshift(img_freq_filted)).real
    # img = np.zeros(img.shape[0]-2,img.shape.[1]-2,3)
    # img_filted  = img_filted[1:-1,1:-1]
    # print(img.shape)
    return img_filted

def freq_filter(img, sigma, gaussian, is_high):
    row = img.shape[0]
    col = img.shape[1]
    # print(img)
    img_rgb = [i.reshape(row, col) for i in np.dsplit(img, 3)]
    # print(img_rgb)
    # img_rgb = [move_center(i) for i in img_rgb]
    # print(img_rgb)
    # plt.imshow(img_rgb[2])
    # plt.draw()
    # plt.pause(1)
    # input("<hit Enter to close>")

    freq_pass_func = make_filter(row, col, sigma, is_high, gaussian)
    img_filted_rgb = [filt(i, freq_pass_func) for i in img_rgb]
    # print(img_filted_rgb)
    # img_filted_rgb = [move_center(i) for i in img_filted_rgb]
    img_filted = np.stack(img_filted_rgb, axis=2)
    # print(img_filted)
    return img_filted.astype('int')

def sampling(img,down):
    row = img.shape[0]
    col = img.shape[1]
    if down:
        resample = np.zeros((row//2, col//2))
        for i in range(row//2):
            for j in range(col//2):
                resample[i][j] = img[i*2][j*2]
    else:
        resample = np.zeros((row*2,col*2))
        for i in range(row*2):
            for j in range(col*2):
                resample[i][j] = img[i//2][j//2]
    return resample

def sample(img, down):
    # nearest method to upsample or downsample
    row = img.shape[0]
    col = img.shape[1]
    img_rgb = [i.reshape(row, col) for i in np.dsplit(img, 3)]
    img_rgb = [sampling(i,down) for i in img_rgb]
    img_resample = np.stack(img_rgb, axis=2)
    return img_resample.astype(int)


def pyramid(img, layers):
    for i in range(layers):
        smooth_img = freq_filter(img, 24, gaussian=True, is_high=False)
        plt.imshow(smooth_img)
        plt.show()
        sub_sample_img = sample(smooth_img,down=True)
        plt.imshow(sub_sample_img)
        plt.show()
        try:
            laplacian_img = img - sample(sub_sample_img,down=False)
        except Exception as e:
            up_sample = sample(sub_sample_img,down=False)
            print(up_sample.shape)
            # plt.imshow(up_sample)
            # plt.show()
            if (img.shape[0]-up_sample.shape[0]) > 0:
                up_sample = np.concatenate((up_sample,[up_sample[-1]]),axis=0)
            print(up_sample.shape)
            if (img.shape[1]-up_sample.shape[1]) > 0:
                print(up_sample[:,-1].shape)
                up_sample = np.concatenate((up_sample,up_sample[:,-1,].reshape(img.shape[0],1,img.shape[2])),axis=1)
            laplacian_img = img - up_sample
        plt.imshow(laplacian_img)
        plt.show()
        img = sub_sample_img

def get_img(img_name):
    img_rgb = imread(IMAGE_DIR + img_name).astype('int')
    return img_rgb


if __name__ == "__main__":
    i = 3
    img_high = get_img(IMAGE_HIGH[i])
    img_low = get_img(IMAGE_LOW[i])

    img_pyramid = pyramid(img_low,5)
    # plt.imshow(img_pyramid)

    plt.show()
    # plt.draw()
    # plt.pause(1)
    # input("<hit Enter to close>")
    # plt.close(plt.figure())
