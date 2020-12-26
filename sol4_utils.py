from scipy.signal import convolve2d
import scipy as si
import numpy as np
import imageio as im
from skimage.color import rgb2gray
GRAY_SCALE=2
RGB=3
MAX_PIXEL=255


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img

def build_gaussian_pyramid(im, max_levels, filter_size):
    row_filter,col_filter=build_filter(filter_size)
    pyr=[im]
    cur=im
    for level in range(max_levels-1):
        G_i=filter_(cur,row_filter,0)
        G_i = filter_(G_i, col_filter,0)
        G_i=G_i[::2,::2]#sample
        if np.shape(G_i)[0]<16 or np.shape(G_i)[1]<16:
            break
        pyr.append(G_i)
        cur=G_i
    return pyr,row_filter

def build_filter(filter_size):
    """
    generate binomial filter vector
    :param size:
    :return: row_filter,col_filter
    """
    base=np.array([1,1])
    filter=base
    for j in range(1,filter_size-1):
        filter=np.convolve(base,filter)
    norm_fact=np.sum(filter)
    filter=filter*(1/norm_fact)
    return np.array(filter.reshape(1,(len(filter)))),np.array(filter.reshape((len(filter),1)))



def filter_(im,filter,expand_or_reduce):
    """
    apply the convolution with the filter
    :param im:
    :param filter:
    :param expand_or_reduce:
    :return: filtered image
    """
    if expand_or_reduce==0:#case of reduction
        filtered_im=si.ndimage.filters.convolve(im,filter)
        #filtered_im=si.signal.convolve2d(im,filter, 'same')
    else:#case of expand
        filter = 2* filter
        filtered_im = si.ndimage.filters.convolve(im,filter)
        #filtered_im = si.signal.convolve2d(im, filter, 'same')
    return filtered_im
def read_image(filename, representation):
    #the function will read an image file and return a normalizes array of
    # its intesitys
    image=im.imread(filename).astype(np.float64)
    if np.amax(image)>1:
        image=image.astype(np.float64)/MAX_PIXEL
    if representation==2 and image.ndim!=GRAY_SCALE:#return RGB from RGB file
        return image
    elif representation==1 and image.ndim==RGB:#return grayscale from RGB file
        return rgb2gray(image)
    elif representation==1 and image.ndim==GRAY_SCALE: #return grayscale from
        # grayscale file
        return image