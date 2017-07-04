# astrometry
Additional classes to astropy to provide support for attitudes, euler angles
etc...
 * astrometry: provide the ExtSkyCoord class (for extended SkyCoord) that
 extends the astropy.coordinates.SkyCoord class by providing three additional
  methods:
  ** angular_distance: a faster implementation than the parent class one,
  using vector calculus
  ** bearing : a fast method to find position angle between ExtSkyCoords




