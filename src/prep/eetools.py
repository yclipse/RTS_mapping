import ee


def gen_roi_geometry(long, lat):
    long1, lat1, long2, lat2 = utils.gen_roi(long, lat)
    geometry = ee.Geometry.Polygon(coords=[[[long1, lat1], [long1, lat2], [long2, lat2], [long2, lat1], ]],
                                   proj='EPSG:4326',
                                   geodesic=None,
                                   maxError=1.,
                                   evenOdd=False
                                   )
    return geometry
