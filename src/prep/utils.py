import numpy as np


def gen_random_offset(offset=0.005):
    '''
    Generate random offset in lat/long degrees
    '''
    return (np.random.random_sample() - np.random.random_sample()) * offset  # (-0.005,0.005)


def gen_roi(long, lat, lenght=0.1):
    '''
    Generate an ROI rectangle with upperleft coord and bottomright coord from a centroid
    '''
    offset = gen_random_offset()

    # (long1,lat1): up left corner (-0.1,0)
    lat1 = lat + offset - length / 2
    long1 = long + (offset - length / 2)  # *2.67   #longitude correction

    # (long2,lat2): bottom right corner (0,0.1)
    lat2 = lat + offset + length / 2
    long2 = long + (offset + length / 2)  # *2.67   #longitude correction

    # print ('diagonal distance (m)=',functions.measure(long1,lat1,long2,lat2))
    # print ('coords1=[',long1,lat1,'],coords2=[',long2,lat2)
    return long1, lat1, long2, lat2


def normalise_uint8(arr):
    '''
    Normalise a nparray to 0-255 in uint8
    '''
    numerator = arr - arr.min()
    denominator = 1 / (arr.max() - arr.min())
    return (numerator * denominator * 255).astype(np.uint8)


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
