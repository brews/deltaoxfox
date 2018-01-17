import numpy as np
from deltaoxfox.modelparams import get_draws


def predict_d18osw(salinity, latlon):
    """Predict d18O of seawater given seawater salinity
    """
    modelparams = get_draws('d18osw')
    alpha, beta, tau2 = modelparams.find_abt2_near(latlon[0], latlon[1])

    y = np.empty((len(salinity), tau2.size))
    for i in tau2:
        alphanow = alpha.iloc[i]
        betanow = beta.iloc[i]
        tau2now = tau2.iloc[i]
        y[:, i] = np.random.normal(salinity * betanow + alphanow,
                                   np.sqrt(tau2now))
    return y


def predict_d18oc(seatemp, d18osw=None, salinity=None, latlon=None):
    """Predict d18O of calcite given seawater temperature and seawater d18O
    """
    assert (d18osw is not None) or ((salinity is not None) and (latlon is not None))
    raise NotImplementedError


def predict_seatemp(d18oc, d18osw=None, salinity=None, latlon=None):
    """Predict seawater temperature given d18O of calcite and seawater d18O
    """
    assert (d18osw is not None) or ((salinity is not None) and (latlon is not None))
    raise NotImplementedError
