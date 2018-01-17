from copy import deepcopy
from pkgutil import get_data
from io import BytesIO
import attr
import pandas as pd

RESOURCE_STR = 'modelparams/bayesreg_{}.h5'


def get_h5_resource(resource, package='deltaoxfox', **kwargs):
    """Read flat HDF5 files as package resources, output for Pandas
    """
    with BytesIO(get_data(package, resource)) as fl:
        data = pd.read_hdf(fl, **kwargs)
    return data


@attr.s
class Draws:
    """Spatially-aware modelparams draws
    """
    alpha = attr.ib()
    beta = attr.ib()
    tau2 = attr.ib()
    latlon = attr.ib()

    def _index_near(self, lat, lon):
        """Get gridpoint index nearest a lat lon
        """
        raise NotImplementedError

    def find_nearest_latlon(self, lat, lon):
        """Find draws gridpoint nearest a given lat lon
        """
        raise NotImplementedError


# Preloading these resources so only need to load them once.
DRAWS = {
    'd18oc': Draws(alpha=get_h5_resource(RESOURCE_STR.format('d18oc'), key='alpha'),
                   beta=get_h5_resource(RESOURCE_STR.format('d18oc'), key='beta'),
                   tau2=get_h5_resource(RESOURCE_STR.format('d18oc'), key='tau2'),
                   latlon=get_h5_resource(RESOURCE_STR.format('d18oc'), key='latlon')),
    'd18osw': Draws(alpha=get_h5_resource(RESOURCE_STR.format('d18osw'), key='alpha'),
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
