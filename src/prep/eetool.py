import ee
import numpy as np
import prep.utils
#


def genRoiGeometry(long, lat, length=0.01, proj='EPSG:4326'):
    '''

    Generate GEE roi geometry polygon using centroid

    '''
    long1, lat1, long2, lat2 = prep.utils.genRoi(long, lat, length)
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


def bandToArray(ee_object, roi, default_value=0):
    '''
    basemap : ee.Image
    roi : ee.Geometry
    default_value(option) : value to replace empty pixels
    Convert single band image to np.array (2d)

    '''

    band_roi = ee_object.sampleRectangle(region=roi, defaultValue=default_value)
    band_arr = band_roi.get(ee_object.bandNames().getInfo()[0])

    return np.array(band_arr.getInfo())


def mdHillshade(DEM,
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


def shadedRelief(slope, hillshade, w_slope=0.5, w_hillshade=0.5):
    '''

    Mosaic hillshade and slope raster to a single image
    which will produce the Shaded Relief, using the ee.ImageCollection.mosaic.

    '''
    sr = ee.ImageCollection([slope.updateMask(w_slope).rename('b1').toFloat(),
                             hillshade.updateMask(w_hillshade).rename('b1').toFloat()
                             ]).mosaic()

    return sr


def relDEM(DEM, kernel_size=15):
    '''

    Calculate local mean of DEM using ee.Image.reduceNeighborhood
    Applies a mean reducer to the neighborhood around each pixel,
    as determined by the given kernel.

    Returns: relative DEM

    '''
    local_mean = DEM.reduceNeighborhood(reducer=ee.Reducer.mean(),
                                        kernel=ee.Kernel.square(kernel_size, 'meters'),
                                        )
    rel_dem = DEM.subtract(local_mean)

    return rel_dem


def bandsToRGB(
    ee_object, bands=None, region=None, properties=None, default_value=None
):
    '''Extracts a rectangular region of pixels from an image into a 2D numpy array per band.
    Args:
        ee_object (object): The image to sample.
        bands (list, optional): The list of band names to extract. Please make sure that all bands have the same spatial resolution. Defaults to None.
        region (object, optional): The region whose projected bounding box is used to sample the image. The maximum number of pixels you can export is 262,144. Resampling and reprojecting all bands to a fixed scale can be useful. Defaults to the footprint in each band.
        properties (list, optional): The properties to copy over from the sampled image. Defaults to all non-system properties.
        default_value (float, optional): A default value used when a sampled pixel is masked or outside a band's footprint. Defaults to None.
    Returns:
        array: 3d numpy array.
    '''
    import numpy as np

    if not isinstance(ee_object, ee.Image):
        print("The input must be an ee.Image.")
        return

    if region is None:
        region = ee_object.geometry()

    try:

        if bands is not None:
            ee_object = ee_object.select(bands)
        else:
            bands = ee_object.bandNames().getInfo()

        band_arrs = ee_object.sampleRectangle(
            region=region, properties=properties, defaultValue=default_value
        )
        band_values = []

        for band in bands:
            band_arr = band_arrs.get(band).getInfo()
            band_value = np.array(band_arr)
            band_values.append(band_value)

        image = np.dstack(band_values)
        return image

    except Exception as e:
        print(e)
