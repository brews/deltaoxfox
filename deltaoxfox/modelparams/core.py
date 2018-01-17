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
        latgridstep = np.asscalar(np.unique(np.diff(self.latlon['lat'].sort_values().unique())))
        longridstep = np.asscalar(np.unique(np.diff(self.latlon['lon'].sort_values().unique())))
        lon_adiff = np.abs(self.latlon['lon'] - lon) <= (longridstep/2)
        lat_adiff = np.abs(self.latlon['lat'] - lat) <= (latgridstep/2)
        return np.where(lon_adiff & lat_adiff)

    def find_nearest_latlon(self, lat, lon):
        """Find draws gridpoints nearest a given lat lon
        """
        idx = self._index_near(lat, lon)
        return self.latlon.iloc[idx].copy()


# Preloading these resources so only need to load them once.
DRAWS = {
    'd18oc': CalciteDraws(alpha=get_h5_resource(RESOURCE_STR.format('d18oc'), key='alpha'),
                          beta=get_h5_resource(RESOURCE_STR.format('d18oc'), key='beta'),
                          tau2=get_h5_resource(RESOURCE_STR.format('d18oc'), key='tau2'),
                          spp_temprange=get_h5_resource(RESOURCE_STR.format('d18oc'),
                                                        key='spp_temprange')),
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
