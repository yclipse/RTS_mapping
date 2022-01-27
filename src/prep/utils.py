import numpy as np


def gen_random_offset(offset=0.005):
    '''
    Generate random offset in lat/long degrees
    '''
    return (np.random.random_sample() - np.random.random_sample()) * offset  # (-0.005,0.005)


def gen_offsetroi(long, lat, lenght=0.1):
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

    #print ('diagonal distance (m)=',functions.measure(long1,lat1,long2,lat2))
    #print ('coords1=[',long1,lat1,'],coords2=[',long2,lat2)
    return long1, lat1, long2, lat2


print('er')
