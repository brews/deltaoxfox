import pytest
import numpy as np
import pandas as pd
from deltaoxfox.predict import (Prediction, predict_d18osw, predict_d18oc,
                                predict_seatemp)


def test_predict_seatemp():
    # TODO(brews): Redo with checked numbers.
    np.random.seed(123)
    d18oc = np.array([-1.71990007])  # Not sure this value is correct.
    goal = np.array([20])
    d18osw = np.array([-0.28134280382815624])
    prior_mean = 30
    prior_std = 20  # seatemp std for prior
    # latlon = (-79.49700165, -18.699981690000016)
    spp = 'ruberwhite'
    victim = predict_seatemp(d18oc=d18oc, prior_mean=prior_mean,
                             prior_std=prior_std, spp=spp, d18osw=d18osw)
    output = victim.ensemble.mean()
    # Note how loose this test is.
    np.testing.assert_allclose(goal, output, atol=1)


def test_predict_seatemp_salinity():
    # TODO(brews): Redo with checked numbers.
    np.random.seed(123)
    goal = 20.300727430496337
    salinity = np.array([34.5])
    latlon = (-79.49700165, -18.699981690000016)
    d18oc = np.array([-1.71990007])
    prior_mean = np.array([20.0])
    prior_std = np.array([10.0])  # seatemp std for prior
    spp = 'ruberwhite'
    victim = predict_seatemp(d18oc=d18oc, prior_mean=prior_mean,
                             prior_std=prior_std, spp=spp, salinity=salinity,
                             latlon=latlon)
    output = victim.ensemble.mean()
    # Note how loose this test is.
    np.testing.assert_allclose(goal, output, atol=1e-1)


def test_predict_d18oc_salinity():
    np.random.seed(123)
    goal = np.array([-1.71990007])
    seatemp = np.array([20])
    salinity = np.array([34.5])
    latlon = (-79.49700165, -18.699981690000016)
    spp = 'ruberwhite'
    victim = predict_d18oc(seatemp=seatemp, spp=spp, salinity=salinity,
                           latlon=latlon)
    output = victim.ensemble.mean()
    # Note how loose this test is.
    np.testing.assert_allclose(goal, output, atol=3e-1)


def test_predict_d18oc_swisotope():
    np.random.seed(123)
    goal = np.array([-1.71990007])  # Not sure this value is correct.
    seatemp = np.array([20])
    d18osw = np.array([-0.28134280382815624])
    # latlon = (-79.49700165, -18.699981690000016)
    spp = 'ruberwhite'
    victim = predict_d18oc(seatemp=seatemp, spp=spp, d18osw=d18osw)
    output = victim.ensemble.mean()
    # Note how loose this test is.
    np.testing.assert_allclose(goal, output, atol=3e-1)


def test_predict_d18osw():
    goal = np.array([-0.28134280382815624])
    latlon = (-79.49700165, -18.699981690000016)
    salinity = np.array([34.5])
    victim = predict_d18osw(salinity=salinity, latlon=latlon)
    output = victim.ensemble.mean()
    np.testing.assert_allclose(goal, output, atol=1e-4)


def test_prediction_precentile():
    goal = np.array([[0, 2, 4], [5, 7, 9]])
    ens = np.reshape(np.arange(10), (2, 5))
    pred = Prediction(ensemble=ens)
    victim = pred.percentile()
    np.testing.assert_equal(victim, goal)
