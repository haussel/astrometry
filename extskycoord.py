import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle, UnitSphericalRepresentation, \
    CartesianRepresentation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

"""
Provides the ExtSkyCoord class, implementing fast distance and bearing 
methods (using cartesian coordinates, and avoiding as much as possible 
trigonometric functions).
"""


class ExtSkyCoord(SkyCoord):
    """
    This class extend the SkyCoord class of astropy.coordinates, providing
    three additional methods.
    """
    def angular_distance(self, positions, collapse=False):
        """
        Compute the angular separation between positions 1 and 2 on the sky.
        Use the vector calculus instead of the Vincenty formula used by astropy
        for the separation method. Even with a code that is not especially
        optimized, the gain in computation time on arrays of coordinates
        is significant.

        Parameters
        ----------
        positions : `astropy.coordinates.SkyCoord`
            second coordinates. Can be as scalar, or contain N2 coordinates
        collapse : boolean.
            If True, and if there is the same number of coordinates in pos1 and
            pos2, then only N1=N2 distances are returned. Defaulted to False.

        Returns
        -------
        output : astropy.coordinates.Angle
            computes the N1xN2 angular distances between coordinates 1 and 2
            (unless collapse is True).
        """

        if not isinstance(positions, SkyCoord):
            raise ValueError("position must be a astropy.coordinates.SkyCoord")

        # cartesian coordinates on the unit sphere (hence passing by
        # UnitSpherical). force scalar input (in the sense of 1 position)
        # into a 3x1 array
        if self.isscalar:
            v1 = self.represent_as(UnitSphericalRepresentation).\
                     to_cartesian().xyz.value[:, np.newaxis]
        else:
            v1 = self.represent_as(UnitSphericalRepresentation).\
                     to_cartesian().xyz.value
        if positions.isscalar:
            v2 = positions.represent_as(UnitSphericalRepresentation).\
                     to_cartesian().xyz.value[:, np.newaxis]
        else:
            v2 = positions.represent_as(UnitSphericalRepresentation).\
                to_cartesian().xyz.value

        # number of points
        nbsc1 = v1.shape[1]
        nbsc2 = v2.shape[1]

        if (nbsc1 == nbsc2) and (collapse is True):
            # dot product of 3xN vectors by 3xN other vectors, N values
            vd = np.einsum('ij,ij->j', v1, v2)
            # norm of cross product of 3xN vectors by 3xN other vectors
            nvc = np.linalg.norm(np.cross(v1, v2, axis=0), axis=0)
        else:
            # dot product of 3xN1  by 3xN2 vectors: N1 x N2 values
            vd = np.einsum('ij,ik->jk', v1, v2)
            #  norm of cross product of 3xN1 vectors by 3xN2 other vectors
            nvc = np.linalg.norm(np.cross(v1[:, :, np.newaxis],
                                          v2[:, np.newaxis, :],
                                          axis=0),
                                 axis=0)

        if self.isscalar and positions.isscalar:
            return Angle(np.arctan2(nvc, vd)[0] * u.rad)
        else:
            return Angle(np.arctan2(nvc, vd) * u.rad)

    def bearing(self, positions, collapse=False):
        """
        Compute the poition angle of positions on the sky with respect to self.
        Even with a code that is not especially optimized, the gain in
        computation time on arrays of coordinates is significant.

        Parameters
        ----------
        positions : `astropy.coordinates.SkyCoord`
            second coordinates. Can be as scalar, or contain N2 coordinates
        collapse : boolean.
            If True, and if there is the same number of coordinates in pos1 and
            pos2, then only N1=N2 distances are returned. Defaulted to False.

        Returns
        -------
        output : astropy.coordinates.Angle
            computes the N1xN2 angular distances between coordinates 1 and 2
            (unless collapse is True).
        """

        if not isinstance(positions, SkyCoord):
            raise ValueError("position must be a astropy.coordinates.SkyCoord")

        # force scalar input (in the sense of 1 position) into a 3x1 array
        if self.isscalar:
            v1 = self.represent_as(UnitSphericalRepresentation).\
                to_cartesian().xyz.value[:, np.newaxis]
        else:
            v1 = self.represent_as(UnitSphericalRepresentation).\
                to_cartesian().xyz.value
        if positions.isscalar:
            v2 = positions.represent_as(UnitSphericalRepresentation).\
                     to_cartesian().xyz.value[:, np.newaxis]
        else:
            v2 = positions.represent_as(UnitSphericalRepresentation).\
                to_cartesian().xyz.value

        # number of points
        nbsc1 = v1.shape[1]
        nbsc2 = v2.shape[1]

        if (nbsc1 == nbsc2) and (collapse is True):
            sine = v2[1, :] * v1[0, :] - v2[0, :] * v1[1, :]
            cosine = v2[2, :] * (v1[0, :] * v1[0, :] + v1[1, :] * v1[1, :]) - \
                     v1[2, :] * (v2[0, :] * v1[0, :] + v2[1, :] * v1[1, :])
        else:
            sine = v2[1, np.newaxis, :] * v1[0, :, np.newaxis] - \
                   v2[0, np.newaxis, :] * v1[1, :, np.newaxis]
            cosine = v2[2, np.newaxis, :] * (v1[0, :, np.newaxis] *
                                             v1[0, :, np.newaxis] +
                                             v1[1, :, np.newaxis] *
                                             v1[1, :, np.newaxis]) - \
                     v1[2, :, np.newaxis] * (v2[0, np.newaxis, :] *
                                             v1[0, :, np.newaxis] +
                                             v2[1, np.newaxis, :] *
                                             v1[1, :, np.newaxis])

        if self.isscalar and positions.isscalar:
            return Angle(np.arctan2(sine, cosine)[0] * u.rad)
        else:
            return Angle(np.arctan2(sine, cosine) * u.rad)

    def tangent_plane(self):
        """

        :return: a tuple, (unit vector(s) along parallel, unit vector(s)
        along meridian)
        """
        sph = self.represent_as(UnitSphericalRepresentation)
        udelta = np.array([-np.sin(sph.lat) * np.cos(sph.lon),
                           -np.sin(sph.lat) * np.sin(sph.lon),
                           np.cos(sph.lat)])
        ualpha = np.array([-np.sin(sph.lon), np.cos(sph.lon), 0])
        return (ualpha, udelta)

    def __getitem__(self, item):
        # Ensure that we obtain a ExtSkyCoord back.
        return ExtSkyCoord(super().__getitem__(item))


