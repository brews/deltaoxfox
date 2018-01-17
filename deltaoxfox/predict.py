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


def predict_d18oc(seatemp, spp, d18osw=None, salinity=None, latlon=None):
    """Predict d18O of calcite given seawater temperature and seawater d18O
    """
    d18oc_modelparams = get_draws('d18oc')
    assert spp in list(d18oc_modelparams.alpha.columns)
    assert (d18osw is not None) or ((salinity is not None) and (latlon is not None))

    d18osw_adj = None
    d18osw_alpha = None
    d18osw_beta = None
    d18osw_tau2 = None

    if d18osw is None:
        modelparams = get_draws('d18osw')
        d18osw_alpha, d18osw_beta, d18osw_tau2 = modelparams.find_abt2_near(latlon[0], latlon[1])
        # We're assuming that model parameter draws for d18osw & d18oc model are
        # the same (for speed)... May change this later with more coding.
        assert d18oc_modelparams.tau2.size == d18osw_tau2.size
    else:
        # Unit adjustment.
        d18osw_adj = d18osw - 0.27

    y = np.empty((len(seatemp), d18oc_modelparams.tau2.size))

    for i in range(d18oc_modelparams.tau2.size):
        tau2_now = d18oc_modelparams.tau2.loc[i, spp]
        beta_now = d18oc_modelparams.beta.loc[i, spp]
        alpha_now = d18oc_modelparams.alpha.loc[i, spp]

        if d18osw is None:
            beta_d18osw_now = d18osw_beta[i]
            alpha_d18osw_now = d18osw_alpha[i]
            tau2_d18osw_now = d18osw_tau2[i]
            d18osw_now = np.random.normal(salinity * beta_d18osw_now + alpha_d18osw_now,
                                          np.sqrt(tau2_d18osw_now))
            d18osw_adj = d18osw_now - 0.27

        y[:, i] = np.random.normal(alpha_now + seatemp * beta_now + d18osw_adj,
                                   np.sqrt(tau2_now))
    return y


def predict_seatemp(d18oc, d18osw=None, salinity=None, latlon=None):
    """Predict seawater temperature given d18O of calcite and seawater d18O
    """
    assert (d18osw is not None) or ((salinity is not None) and (latlon is not None))
    raise NotImplementedError