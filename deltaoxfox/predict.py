import numpy as np
import attr
from deltaoxfox.modelparams import get_draws


@attr.s()
class Prediction:
    ensemble = attr.ib()

    def percentile(self, q=None, interpolation='nearest'):
        """Compute the qth ranked percentile from ensemble members.

        Parameters
        ----------
        q : float ,sequence of floats, or None, optional
            Percentiles (i.e. [0, 100]) to compute. Default is 5%, 50%, 95%.
        interpolation : str, optional
            Passed to numpy.percentile. Default is 'nearest'.

        Returns
        -------
        perc : ndarray
            A 2d (nxm) array of floats where n is the number of predictands in
            the ensemble and m is the number of percentiles ('len(q)').
        """
        if q is None:
            q = [5, 50, 95]
        q = np.array(q, dtype=np.float64, copy=True)

        # Because analog ensembles have 3 dims
        target_axis = list(range(self.ensemble.ndim))[1:]

        perc = np.percentile(self.ensemble, q=q, axis=target_axis,
                             interpolation=interpolation)
        return perc.T


def predict_d18osw(salinity, latlon):
    """Predict d18O of seawater given seawater salinity
    """
    modelparams = get_draws('d18osw')
    alpha, beta, tau2 = modelparams.find_abt2_near(latlon[0], latlon[1])

    y = np.empty((len(salinity), len(tau2)))
    y[:] = np.nan
    for i, tau2now in enumerate(tau2):
        alphanow = alpha.iloc[i]
        betanow = beta.iloc[i]
        y[:, i] = np.random.normal(salinity * betanow + alphanow,
                                   np.sqrt(tau2now))
    return Prediction(ensemble=y)


def predict_d18oc(seatemp, spp, d18osw=None, salinity=None, latlon=None):
    """Predict d18O of calcite given seawater temperature and seawater d18O
    """
    d18oc_modelparams = get_draws('d18oc')
    assert spp in list(d18oc_modelparams.alpha.columns)
    assert (d18osw is not None) or ((salinity is not None) and (latlon is not None))

    d18oc_tau2 = d18oc_modelparams.tau2.loc[:, spp].copy()
    d18oc_beta = d18oc_modelparams.beta.loc[:, spp].copy()
    d18oc_alpha = d18oc_modelparams.alpha.loc[:, spp].copy()

    d18osw_adj = None
    d18osw_alpha = None
    d18osw_beta = None
    d18osw_tau2 = None

    if d18osw is None:
        modelparams = get_draws('d18osw')
        d18osw_alpha, d18osw_beta, d18osw_tau2 = modelparams.find_abt2_near(latlon[0], latlon[1])
        # We're assuming that model parameter draws for d18osw & d18oc model are
        # the same (for speed)... May change this later with more coding.
        assert d18oc_modelparams.tau2.shape[0] == d18osw_tau2.shape[0]
    else:
        # Unit adjustment.
        d18osw_adj = d18osw - 0.27

    y = np.empty((len(seatemp), len(d18oc_modelparams.tau2)))
    y[:] = np.nan

    for i, tau2_now in enumerate(d18oc_tau2):
        beta_now = d18oc_beta.iloc[i]
        alpha_now = d18oc_alpha.iloc[i]

        if d18osw is None:
            beta_d18osw_now = d18osw_beta.iloc[i]
            alpha_d18osw_now = d18osw_alpha.iloc[i]
            tau2_d18osw_now = d18osw_tau2.iloc[i]
            d18osw_now = np.random.normal(salinity * beta_d18osw_now + alpha_d18osw_now,
                                          np.sqrt(tau2_d18osw_now))
            d18osw_adj = d18osw_now - 0.27
        mu = alpha_now + seatemp * beta_now + d18osw_adj
        y[:, i] = np.random.normal(mu, np.sqrt(tau2_now))
    return Prediction(ensemble=y)


def predict_seatemp(d18oc, spp, prior_std, d18osw=None, salinity=None, latlon=None):
    """Predict seawater temperature given d18O of calcite and seawater d18O
    """
    d18oc_modelparams = get_draws('d18oc')
    assert spp in list(d18oc_modelparams.alpha.columns)
    assert (d18osw is not None) or ((salinity is not None) and (latlon is not None))

    d18oc_tau2 = d18oc_modelparams.tau2.loc[:, spp].copy()
    d18oc_beta = d18oc_modelparams.beta.loc[:, spp].copy()
    d18oc_alpha = d18oc_modelparams.alpha.loc[:, spp].copy()

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

    y = np.empty((len(d18oc), len(d18oc_modelparams.tau2)))
    y[:] = np.nan

    for i, tau2_now in enumerate(d18oc_tau2):
        beta_now = d18oc_beta.iloc[i]
        alpha_now = d18oc_alpha.iloc[i]

        if d18osw is None:
            beta_d18osw_now = d18osw_beta.iloc[i]
            alpha_d18osw_now = d18osw_alpha.iloc[i]
            tau2_d18osw_now = d18osw_tau2.iloc[i]
            d18osw_now = np.random.normal(salinity * beta_d18osw_now + alpha_d18osw_now,
                                          np.sqrt(tau2_d18osw_now))
            d18osw_adj = d18osw_now - 0.27

        mu = -((alpha_now + d18osw_adj + np.sqrt(tau2_now) - d18oc) / beta_now)
        y[:, i] = np.random.normal(mu, prior_std)
    return Prediction(ensemble=y)
