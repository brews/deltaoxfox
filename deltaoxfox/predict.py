import numpy as np
from numba import jit
import attr
import bayfox as bfox
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


@jit
def predict_d18osw(salinity, latlon):
    """Predict d18O of seawater given seawater salinity
    """
    salinity = np.array(salinity)

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


@jit
def predict_d18oc(seatemp, spp=None, d18osw=None, salinity=None, latlon=None):
    """Predict d18O of calcite given seawater temperature and seawater d18O
    """
    seatemp = np.array(seatemp)
    if d18osw is not None:
        d18osw = np.array(d18osw)
    if salinity is not None:
        salinity = np.array(salinity)

    assert (d18osw is not None) or ((salinity is not None) and (latlon is not None)), "need 'd18osw' or 'salinity' and 'latlon'"

    # Get our draws for d18oc prediction form bayfox.
    # Hack to get around the difference in sample size between d18Oc (bayfox)
    # and d18Osw MCMC parameter draw size.
    n = 5000
    d18oc_alpha, d18oc_beta, d18oc_tau = bfox.modelparams.get_draws(foram=spp, seasonal_seatemp=False)
    d18oc_alpha = np.random.choice(d18oc_alpha, n)
    d18oc_beta = np.random.choice(d18oc_beta, n)
    d18oc_tau = np.random.choice(d18oc_tau, n)

    # This is all clunky and could use a cleanup.

    d18osw_adj = None
    d18osw_alpha = None
    d18osw_beta = None
    d18osw_tau2 = None

    if d18osw is None:
        modelparams = get_draws('d18osw')
        d18osw_alpha, d18osw_beta, d18osw_tau2 = modelparams.find_abt2_near(latlon[0], latlon[1])
        # We're assuming that model parameter draws for d18osw & d18oc model are
        # the same (for speed)... May change this later with more coding.
        assert d18oc_tau.shape[0] == d18osw_tau2.shape[0]
    else:
        # Unit adjustment.
        d18osw_adj = d18osw - 0.27

    y = np.empty((len(seatemp), len(d18oc_tau)))
    y[:] = np.nan

    for i, tau_now in enumerate(d18oc_tau):
        beta_now = d18oc_beta[i]
        alpha_now = d18oc_alpha[i]

        if d18osw is None:
            beta_d18osw_now = d18osw_beta.iloc[i]
            alpha_d18osw_now = d18osw_alpha.iloc[i]
            tau2_d18osw_now = d18osw_tau2.iloc[i]
            d18osw_now = np.random.normal(salinity * beta_d18osw_now + alpha_d18osw_now,
                                          np.sqrt(tau2_d18osw_now))
            d18osw_adj = d18osw_now - 0.27
        mu = alpha_now + seatemp * beta_now + d18osw_adj
        y[:, i] = np.random.normal(mu, tau_now)
    return Prediction(ensemble=y)


@jit
def predict_seatemp(d18oc,  prior_mean, prior_std, spp=None, d18osw=None, salinity=None, latlon=None):
    """Predict seawater temperature given d18O of calcite and seawater d18O
    """
    d18oc = np.array(d18oc)
    prior_mean = np.array(prior_mean)
    prior_std = np.array(prior_std)
    if d18osw is not None:
        d18osw = np.array(d18osw)
    if salinity is not None:
        salinity = np.array(salinity)

    assert (d18osw is not None) or ((salinity is not None) and (latlon is not None)), "need 'd18osw' or 'salinity' and 'latlon'"

    # Get our draws for d18oc prediction form bayfox.
    # Hack to get around the difference in sample size between d18Oc (bayfox)
    # and d18Osw MCMC parameter draw size.
    n = 5000
    d18oc_alpha, d18oc_beta, d18oc_tau = bfox.modelparams.get_draws(foram=spp, seasonal_seatemp=False)
    d18oc_alpha = np.random.choice(d18oc_alpha, n)
    d18oc_beta = np.random.choice(d18oc_beta, n)
    d18oc_tau2 = np.random.choice(d18oc_tau, n) ** 2

    d18osw_adj = None
    d18osw_alpha = None
    d18osw_beta = None
    d18osw_tau2 = None

    if d18osw is None:
        modelparams = get_draws('d18osw')
        d18osw_alpha, d18osw_beta, d18osw_tau2 = modelparams.find_abt2_near(latlon[0], latlon[1])
        # We're assuming that model parameter draws for d18osw & d18oc model are
        # the same (for speed)... May change this later with more coding.
        assert d18oc_tau2.shape[0] == d18osw_tau2.shape[0]
    else:
        # Unit adjustment.
        d18osw_adj = d18osw - 0.27

    nd = len(d18oc)
    prior_par = {'mu': np.ones(nd) * prior_mean,
                 'inv_cov': np.eye(nd) * prior_std ** -2}

    y = np.empty((len(d18oc), len(d18oc_tau2)))
    y[:] = np.nan

    for i, tau2_now in enumerate(d18oc_tau2):
        beta_now = d18oc_beta[i]
        alpha_now = d18oc_alpha[i]

        if d18osw is None:
            beta_d18osw_now = d18osw_beta.iloc[i]
            alpha_d18osw_now = d18osw_alpha.iloc[i]
            tau2_d18osw_now = d18osw_tau2.iloc[i]
            d18osw_now = np.random.normal(salinity * beta_d18osw_now + alpha_d18osw_now,
                                          np.sqrt(tau2_d18osw_now))
            d18osw_adj = d18osw_now - 0.27

        y[:, i] = _target_timeseries_pred(d18osw_now=d18osw_adj,
                                          alpha_now=alpha_now,
                                          beta_now=beta_now,
                                          tau2_now=tau2_now,
                                          proxy_ts=d18oc,
                                          prior_pars=prior_par)
    return Prediction(ensemble=y)


@jit
def _target_timeseries_pred(d18osw_now, alpha_now, beta_now, tau2_now, proxy_ts, prior_pars):
    """

    Parameters
    ----------
    alpha_now : scalar
    beta_now : scalar
    tau2_now : scalar
        Current value of the residual variance.
    proxy_ts : ndarray
        Time series of the proxy. Assume no temporal structure for now, as
        timing is not equal.
    prior_pars : dict
        mu : ndarray
            Prior means for each element of the time series.
        inv_cov: ndarray
            Inverse of the prior covariance matrix for the time series.

    Returns
    -------
    Sample of target time series vector conditional on the rest.
    """
    # TODO(brews): Above docstring is based on original MATLAB. Needs cleanup.
    n_ts = len(proxy_ts)

    # Inverse posterior covariance matrix
    inv_post_cov = prior_pars['inv_cov'] + beta_now ** 2 / tau2_now * np.eye(n_ts)

    # Used cholesky to speed things up!
    post_cov = np.linalg.solve(inv_post_cov, np.eye(n_ts))
    sqrt_post_cov = np.linalg.cholesky(post_cov).T
    # Get first factor for the mean
    mean_first_factor = (prior_pars['inv_cov'] @ prior_pars['mu'] + (1/tau2_now)
                         * beta_now * (proxy_ts - alpha_now - d18osw_now))
    mean_full = post_cov @ mean_first_factor

    timeseries_pred = mean_full + sqrt_post_cov @ np.random.randn(n_ts).T

    return timeseries_pred
