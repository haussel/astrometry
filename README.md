# astrometry
Additional classes to astropy to provide support for attitudes, euler angles
etc...

## astrometry.py

Provides the ExtSkyCoord class (for extended SkyCoord) that extends the
astropy.coordinates.SkyCoord class by providing three additional methods:

- angular_distance(): a faster implementation than the parent class one,
  using vector calculus
- bearing() : a fast method to find position angle between ExtSkyCoords, again
based vector calculus
- tangent_plane(): returns a tuple of tangent unit vectors, the first  along 
the parallel, the second along the meridian
 
 





