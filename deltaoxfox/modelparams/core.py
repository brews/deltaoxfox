import os
from copy import deepcopy
import attr
import numpy as np
import pandas as pd


RESOURCE_STR = 'bayesreg_{}.h5'


def get_h5_resource(fl, **kwargs):
    """Read flat HDF5 files as package resources, output for Pandas
    """
    here = os.path.abspath(os.path.dirname(__file__))
    flpath = os.path.join(here, fl)
    data = pd.read_hdf(flpath, **kwargs)
    return data


def chord_distance(latlon1, latlon2):
    """Chordal distance between two sequences of (lat, lon) points

    Parameters
    ----------
    latlon1 : sequence of tuples
        (latitude, longitude) for one set of points.
    latlon2 : sequence of tuples
        A sequence of (latitude, longitude) for another set of points.

    Returns
    -------
    dists : 2d array
        An mxn array of Earth chordal distances [1]_ (km) between points in
        latlon1 and latlon2.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chord_(geometry)

    """
    earth_radius = 6378.137  # in km

    latlon1 = np.atleast_2d(latlon1)
    latlon2 = np.atleast_2d(latlon2)

    n = latlon1.shape[0]
    m = latlon2.shape[0]

    paired = np.hstack((np.kron(latlon1, np.ones((m, 1))),
                        np.kron(np.ones((n, 1)), latlon2)))

    latdif = np.deg2rad(paired[:, 0] - paired[:, 2])
    londif = np.deg2rad(paired[:, 1] - paired[:, 3])

    a = np.sin(latdif / 2) ** 2
    b = np.cos(np.deg2rad(paired[:, 0]))
    c = np.cos(np.deg2rad(paired[:, 2]))
    d = np.sin(np.abs(londif) / 2) ** 2

    half_angles = np.arcsin(np.sqrt(a + b * c * d))

    dists = 2 * earth_radius * np.sin(half_angles)

    return dists.reshape(m, n)


@attr.s
class Draws:
    """Model parameters draws
    """
    alpha = attr.ib()
    beta = attr.ib()
    tau2 = attr.ib()


@attr.s
class CalciteDraws(Draws):
    """Model parameters draws for Calcite in different spp.
    """
    spp_temprange = attr.ib()
    spp_d18oswrange = attr.ib()


@attr.s
class SeawaterDraws(Draws):
    """Spatially-aware model parameters draws
    """
    latlon = attr.ib()

    def _index_near(self, lat, lon):
        """Get gridpoint index nearest a lat lon
        """
        assert -90 <= lat <= 90
        assert -180 < lon <= 180
        d = chord_distance((lat, lon), self.latlon)
        return np.where((d == np.min(d)).ravel())

    def find_nearest_latlon(self, lat, lon):
        """Find draws gridpoints nearest a given lat lon
        """
        idx = self._index_near(lat, lon)
        return self.latlon.iloc[idx].copy()

    def find_abt2_near(self, lat, lon):
        """Find alpha, beta, tau2 at the nearest valid grid to a lat lon point
        """
        idx = self._index_near(lat, lon)
        alpha = self.alpha.iloc[idx].copy().squeeze()
        beta = self.beta.iloc[idx].copy().squeeze()
        tau2 = self.tau2.iloc[idx].copy().squeeze()
        return alpha, beta, tau2


# Preloading these resources so only need to load them once.
DRAWS = {
    'd18oc': CalciteDraws(alpha=get_h5_resource(RESOURCE_STR.format('d18oc'), key='alpha'),
                          beta=get_h5_resource(RESOURCE_STR.format('d18oc'), key='beta'),
                          tau2=get_h5_resource(RESOURCE_STR.format('d18oc'), key='tau2'),
                          spp_temprange=get_h5_resource(RESOURCE_STR.format('d18oc'),
                                                        key='spp_temprange'),
                          spp_d18oswrange=get_h5_resource(RESOURCE_STR.format('d18oc'),
                                                          key='spp_d18oswrange')
                          ),
    'd18osw': SeawaterDraws(alpha=get_h5_resource(RESOURCE_STR.format('d18osw'), key='alpha'),
                            beta=get_h5_resource(RESOURCE_STR.format('d18osw'), key='beta'),
                            tau2=get_h5_resource(RESOURCE_STR.format('d18osw'), key='tau2'),
                            latlon=get_h5_resource(RESOURCE_STR.format('d18osw'), key='latlon')),
}


def get_draws(drawtype):
    """Get modelparam Draws instance for draw type
    """
    drawtype = drawtype.lower()
    assert drawtype in ['d18oc', 'd18osw']
    return deepcopy(DRAWS[drawtype])
