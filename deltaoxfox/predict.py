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
    """Predict seawater δ18O given seawater salinity and location.

    Parameters
    ----------
    salinity : array_like or scalar
        Seawater salinity.
    latlon : tuple[float]
        (latitude, longitude) tuple for the prediction site.

    Returns
    -------
    out : Prediction
        Model prediction of site's seawater δ18O.
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
def predict_d18oc(seatemp, spp=None, d18osw=None, salinity=None, latlon=None, sesonal_seatemp=False):
    """Predict δ18O of foram calcite given seawater temp and seawater δ18O or salinity

    Estimates seawater δ18O from ``salinity`` and ``latlon`` if ``d18osw`` not
    given.

    Parameters
    ----------
    seatemp : array_like or scalar
        n-length array or scalar of sea-surface temperature (°C).
    spp : str, optional
        Foraminifera group name of ``d18oc`` sample. Can be 'G. ruber pink',
        'G. ruber white', 'G. sacculifer', 'N. pachyderma sinistral',
        'G. bulloides', 'N. incompta' or ``None``. If ``None``, pooled
        calibration model is used.
    d18osw : array_like or scalar or None, optional
        n-length array or scalar of δ18O of seawater (‰; VSMOW). If not scalar,
        must be the same length as ``d18oc``. If not given, must give values for
        ``salinity`` and ``latlon``.
    salinity : array_like or scalar or None, optional
        n-length array or scalar of sea water salinity values. If not scalar,
        must be the same length as ``d18oc``. Must either define ``d18osw`` or
        ``salinity`` and ``latlon``.
    latlon : tuple[float], optional
        (latitude, longitude) for the prediction site. Must either define
        ``d18osw`` or ``salinity`` and ``latlon``.
    seasonal_seatemp : bool, optional
        Indicates whether sea-surface temperature is annual or seasonal
        estimate. If ``True``, ``foram`` must be specified.


    Returns
    -------
    out : Prediction
        Model prediction giving estimated δ18O of planktic foraminiferal calcite
        (‰; VPDB).
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

    # For legacy DA, we need to normalize species names.
    if str(spp) in ['T. sacculifer', 'G. sacculifer']:
        spp = 'T. sacculifer'
    elif str(spp) in ['G. ruber pink', 'G. ruber white', 'G. ruber']:
        spp = 'G. ruber'
    elif str(spp) in ['N. pachyderma sinistral', 'N. pachyderma']:
        spp = 'N. pachyderma'
    d18oc_alpha, d18oc_beta, d18oc_tau = bfox.modelparams.get_draws(foram=spp, seasonal_seatemp=sesonal_seatemp)
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
