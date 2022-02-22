import ee
import numpy as np
import prep.utils
#


def gen_roi_geometry(long, lat, length=0.01, proj='EPSG:4326'):
    '''

    Generate GEE roi geometry polygon using centroid

    '''
    long1, lat1, long2, lat2 = prep.utils.gen_roi(long, lat, length)
    geometry = ee.Geometry.Polygon(coords=[[[long1, lat1],
                                            [long1, lat2],
                                            [long2, lat2],
                                            [long2, lat1], ]],
                                   proj=proj,
                                   geodesic=None,
                                   maxError=1.,
                                   evenOdd=False
                                   )
    return geometry


def band_to_arr(basemap, band: str, roi):
    '''
    basemap:ee.Image
    roi:ee.Geometry

    Convert selected band to np.array (2d)
    '''
    band_roi = basemap.sampleRectangle(region=roi, defaultValue=0)
    band_arr = band_roi.get(band)
    return np.array(band_arr.getInfo())


def multidirectional_hillshade(DEM,
                               w_n=0.125,
                               w_ne=0.125,
                               w_e=0.125,
                               w_se=0.125,
                               w_s=0.125,
                               w_sw=0.125,
                               w_w=0.125,
                               w_nw=0.125):
    '''

    Using eight different traditional hillshades (input,azimuth,altitude).multiply(weight),
    to produce a weighted multi-directional hillshade.

    '''
    N = ee.Terrain.hillshade(DEM, 0, 36).multiply(w_n)
    NE = ee.Terrain.hillshade(DEM, 45, 44).multiply(w_ne)
    E = ee.Terrain.hillshade(DEM, 90, 56).multiply(w_e)
    SE = ee.Terrain.hillshade(DEM, 135, 68).multiply(w_se)
    S = ee.Terrain.hillshade(DEM, 180, 80).multiply(w_s)
    SW = ee.Terrain.hillshade(DEM, 225, 68).multiply(w_sw)
    W = ee.Terrain.hillshade(DEM, 270, 56).multiply(w_w)
    NW = ee.Terrain.hillshade(DEM, 315, 44).multiply(w_nw)

    multi_hillshade = N.add(NE).add(E).add(SE).add(S).add(SW).add(W).add(NW)

    return multi_hillshade


def shaded_relief(slope, hillshade, w_slope=0.5, w_hillshade=0.5):
    '''

    Mosaic hillshade and slope raster to a single image
    which will produce the Shaded Relief, using the ee.ImageCollection.mosaic.

    '''
    sr = ee.ImageCollection([slope.updateMask(w_slope).rename('b1').toFloat(),
                             hillshade.updateMask(w_hillshade).rename('b1').toFloat()
                             ]).mosaic()

    return sr


def rel_dem(DEM, kernel_size=15):
    '''

    Calculate local mean of DEM using ee.Image.reduceNeighborhood
    Applies a mean reducer to the neighborhood around each pixel,
    as determined by the given kernel.

    return: relative DEM

    '''
    local_mean = DEM.reduceNeighborhood(reducer=ee.Reducer.mean(),
                                        kernel=ee.Kernel.square(kernel_size, 'meters'),
                                        )
    rel_dem = DEM.subtract(local_mean)

    return rel_dem
