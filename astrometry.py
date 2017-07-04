import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle, UnitSphericalRepresentation, \
    CartesianRepresentation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


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


# This is to plot nice arrows
class Arrow3D(FancyArrowPatch):
    """

    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plotradec(ra, dec, fig, viewaz = -45, viewel=30., pa = None):
    """

    :param ra:
    :param dec:
    :param fig:
    :param viewaz:
    :param viewel:
    :param pa:
    :return:
    """
    sx = ESkyCoord(ra=0 * u.deg, dec=0*u.deg, frame='icrs')
    sy = ESkyCoord(ra=90 * u.deg, dec=0*u.deg, frame='icrs')
    sz = ESkyCoord(ra=0 * u.deg, dec=90*u.deg, frame='icrs')
    sp = ESkyCoord(ra=ra, dec=dec, frame='icrs')
    s = ESkyCoord([sx, sy, sz, sp])
    xyz = s.cartesian

    ax = fig.gca(projection='3d')

    ax.set_aspect("equal")


    # Draw the unit vectors and the object and label them
    for i in range(3):
        a = Arrow3D([0, xyz.x[i]], [0, xyz.y[i]], [0, xyz.z[i]], mutation_scale=20,
                    lw=3, arrowstyle="->", color="k")
        ax.add_artist(a)
    ax.text(xyz.x[0], xyz.y[0], xyz.z[0], r'$+\vec{x}$',size=20)
    ax.text(xyz.x[1], xyz.y[1], xyz.z[1], r'$+\vec{y}$',size=20)
    ax.text(xyz.x[2], xyz.y[2], xyz.z[2], r'$+\vec{z}$',size=20)
    ax.text(xyz.x[3], xyz.y[3], xyz.z[3], r'$+\vec{v}$',size=20)
    a = Arrow3D([0, xyz.x[3]], [0, xyz.y[3]], [0, xyz.z[3]], mutation_scale=20,
                lw=3, arrowstyle="->", color="r")
    ax.add_artist(a)
    ax.scatter(xyz.x, xyz.y, xyz.z, 'g')

    # plot the surface of a sphere
    uu = np.linspace(0, 2 * np.pi, 100)
    vv = np.linspace(0, np.pi, 100)
    xx = np.outer(np.cos(uu), np.sin(vv))
    yy = np.outer(np.sin(uu), np.sin(vv))
    zz = np.outer(np.ones(np.size(uu)), np.cos(vv))

    # Plot the surface
    ax.plot_surface(xx, yy, zz, color='b', alpha=0.1)
    ax.plot_surface(xx, yy, np.zeros(xx.shape), color='k', alpha=0.2)

    theta = np.arange(0, 361) * u.deg
    xx = np.cos(theta)
    yy = np.sin(theta)

    ax.plot(xx,yy, 0.,  'k:')
    ax.plot(np.zeros(361), xx, yy,  'k:')
    ax.plot(xx, np.zeros(361), yy,  'k:')

    if dec < 0:
        decplot = np.arange(0, dec.value, -1)*u.deg
    else:
        decplot = np.arange(0, dec.value, 1)*u.deg

    raplot = np.ones(decplot.shape) * ra

    splot = ESkyCoord(ra=raplot, dec=decplot, frame='icrs').cartesian

    ax.plot(splot.x, splot.y, splot.z, ':r')
    ax.plot([splot.x[0], 0], [splot.y[0], 0], [splot.z[0], 0], ':r')
    b = Arrow3D([splot.x[-2],splot.x[-1]],  [splot.y[-2], splot.y[-1]],
                [splot.z[-2], splot.z[-1]], color='r', mutation_scale=20,
                lw=3, arrowstyle="->")
    ax.add_artist(b)
    nb = int(len(splot.x)/2)
    ax.text(splot.x[nb], splot.y[nb], splot.z[nb], r'$\delta$', size=20, color='r')

    raplot = np.arange(0, ra.degree) * u.deg
    decplot = raplot * 0.
    splot = SkyCoord(ra=raplot, dec=decplot, frame='icrs').cartesian
    ax.plot(splot.x, splot.y, splot.z, ':r')
    b = Arrow3D([splot.x[-2],splot.x[-1]],  [splot.y[-2], splot.y[-1]],
                [splot.z[-2], splot.z[-1]], color='r', mutation_scale=20,
                lw=3, arrowstyle="->")
    ax.add_artist(b)
    nb = int(len(splot.x)/2)
    ax.text(splot.x[nb], splot.y[nb], splot.z[nb], r'$\alpha$', size=20, color='r')
    ax.text(0.1, 0.1, 1.2, 'North', size=20)
    ax.text(0.1, 1.4, 0.1, 'East', size=20)

    if pa is not None:
        # unit vectors on the plane tangential to the source:
        ualpha , udelta = sp.tangent_plane()
        offset = 0.3 * (ualpha[np.newaxis, :] * np.cos(theta[:, np.newaxis]) +
                        udelta[np.newaxis, :] * np.sin(theta[:, np.newaxis]))
        allp = xyz[3].xyz + offset
        sallp = CartesianRepresentation(x=allp, xyz_axis=1)
        spc = SkyCoord(sallp.represent_as(UnitSphericalRepresentation), frame='icrs')
        paa = sp.position_angle(spc)
        ii = np.argmin(np.abs(paa - pa))
        xyzpa = spc[ii].cartesian
        c = Arrow3D([xyz[3].x, xyzpa.x], [xyz[3].y, xyzpa.y], [xyz[3].z, xyzpa.z], mutation_scale=20,
                    lw=3, arrowstyle="->", color="g")
        ax.add_artist(c)
        ii0 = np.argmin(np.abs(paa))
        xyzpa0 = spc[ii0].cartesian
        ax.plot([xyz[3].x, xyzpa0.x], [xyz[3].y, xyzpa0.y], [xyz[3].z, xyzpa0.z], 'g--')
        xyzpp = spc.cartesian
        if ii0 > 0:
            ax.plot(xyzpp[ii:ii0].x, xyzpp[ii:ii0].y, xyzpp[ii:ii0].z, 'g--')
        else:
            ax.plot(xyzpp[ii0:ii].x, xyzpp[ii0:ii].y, xyzpp[ii0:ii].z, 'g--')
        imid = int((ii + ii0) / 2)
        ax.text(xyzpp[imid].x, xyzpp[imid].y, xyzpp[imid].z, 'PA', color='g', size=20)

    ax.view_init(elev=viewel, azim=viewaz)

    return ax

