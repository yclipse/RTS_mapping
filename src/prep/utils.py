import numpy as np
import random


# def gen_random_offset(offset=0.005):
#     '''
#     Generate random offset in lat/long degrees
#     '''
#     return (np.random.random_sample() - np.random.random_sample()) * offset  # (-0.005,0.005)


def gen_roi(lon, lat, length=0.01):
    '''
    Generate an ROI rectangle with upperleft coord and bottomright coord from a centroid
    length unit in deg
    '''
    offset = random.uniform(-length / 2, length / 2)

    # (long1,lat1): up left corner (-0.01,0)
    lat1 = lat + offset - length / 2
    lon1 = lon + offset - length / 2  # *2.67   #longitude correction

    # (long2,lat2): bottom right corner (0,0.01)
    lat2 = lat + offset + length / 2
    lon2 = lon + offset + length / 2  # *2.67   #longitude correction

    # print ('diagonal distance (m)=',functions.measure(long1,lat1,long2,lat2))
    # print ('coords1=[',long1,lat1,'],coords2=[',long2,lat2)
    return lon1, lat1, lon2, lat2


def normalise(arr):
    '''
    Normalise an np.array to 0-1
    '''
    numerator = arr - arr.min()
    denominator = arr.max() - arr.min()
    return (numerator / denominator)


def pad_arr(img, side_len=280):
    '''
    pad nparrays to a fixed shape (square)
    '''
    if img.ndim == 3:  # if the image is rgb
        # print ('padding an rgb image')
        shape = np.zeros((side_len, side_len, 3))
        for channel in range(img.shape[-1]):
            shape[:img.shape[0], :img.shape[1], channel] = img[..., channel]
    else:  # 1 channel - label
        # print ('padding a label image')
        shape = np.zeros((side_len, side_len))
        shape[:img.shape[0], :img.shape[1]] = img
    return shape


def reshape_labels(ls):
    '''
    convert numerical labels to one-hot encoded labels
    '''
    out = np.zeros((ls.shape[0], ls.shape[1], ls.shape[2], 2))  # split background and forground
    for i in range(ls.shape[0]):
        out[i, ..., 0] = ~ls[i, ..., -1]  # not label
        out[i, ..., 1] = ls[i, ..., -1]  # label
    return out.astype('bool')


def dstack_imgs(*arr):
    '''

    stack N 2d images of to 1 ndarray using np.dstack
    for example N images of shape (W,H) will be stacked as a (W,H,N) ndarray

    '''
    for i in arr:
        i = np.expand_dims(i, -1)

    stacked_arr = np.dstack(*arr)

    return stacked_arr
