import numpy as np
import random

# def gen_random_offset(offset=0.005):
#     '''
#     Generate random offset in lat/long degrees
#     '''
#     return (np.random.random_sample() - np.random.random_sample()) * offset  # (-0.005,0.005)


def genRoi(lon, lat, length=0.01, w_offset=0.5):
    '''
    Generate an ROI rectangle with upperleft coord and bottomright coord from a centroid
    length unit in deg

    '''
    offset = random.uniform(-length / 2, length / 2) * w_offset

    # (long1,lat1): up left corner (-0.01,0)
    lat1 = lat + offset - length / 2
    lon1 = lon + offset - length / 2  # *2.67   #longitude correction

    # (long2,lat2): bottom right corner (0,0.01)
    lat2 = lat + offset + length / 2
    lon2 = lon + offset + length / 2  # *2.67   #longitude correction

    return lon1, lat1, lon2, lat2


def normalise(arr):
    '''
    Normalise an np.array to 0-1

    '''
    numerator = arr - arr.min()
    denominator = arr.max() - arr.min()
    return (numerator / denominator)

def zero_centering_norm(img, axis=(0, 1), c=1e-8):
    """
    Normalize to zero mean and unit standard deviation along the given axis.
    Args:
        img (numpy or cupy): array (w, h, c)
        axis (integer tuple): into or tuple of width and height axis
        c (float): epsilon to bound given std value
    Return:
        Normalize single image
    ----------
    Example
    ----------
        image_normalize(arr, axis=(0, 1), c=1e-8)
    """
    return (img - img.mean(axis)) / (img.std(axis) + c)


def padArr(img, side_len=280):
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


def oneHotLabels(lb):
    '''
    convert binary labels to one-hot encoded labels
    work with 3d array (W,H,C)
    '''
    out = np.zeros((lb.shape[0], lb.shape[1], 2))  # split background and forground

    out[..., 0] = ~lb.astype(bool)  # not label
    out[..., 1] = lb.astype(bool)  # label
    return out


def CropAndPad(arr, target_size):
    '''
    resize a 3d array to a fixed size with crop and pad

    '''
    def f(a): return (abs(a)+a)/2  # negative values become zero

    hpad = f(target_size[0]-arr.shape[0])
    wpad = f(target_size[1]-arr.shape[1])

    pad = np.pad(arr, ((0, int(hpad)), (0, int(wpad)), (0, 0)))
    out = pad[:target_size[0], :target_size[1], :]
    return out


def vstack_list(lst):
    'vstack a list'
    for i in range(len(lst)):
        lst[i] = np.expand_dims(lst[i], 0)
    out = np.vstack(lst)
    return out


def rgb2gray(rgb):
    '''convert rgb img array to grayscale'''
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def flatten_list(t):
    'flatten a list of numbers'
    return [item for sublist in t for item in sublist]


def flatten_str_list(A):
    'flatten a list and strings wont split'
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt
