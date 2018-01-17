import pytest
import numpy as np
import pandas as pd
from deltaoxfox.predict import Prediction, predict_d18osw


def test_predict_d18osw():
    # TODO(brews): Finish test.
    latlon = (42, -125.2)
    salinity = np.arange(30)
    victim = predict_d18osw(salinity, latlon)


def test_prediction_precentile():
    goal = np.array([[0, 2, 4], [5, 7, 9]])
    ens = np.reshape(np.arange(10), (2, 5))
    pred = Prediction(ensemble=ens)
    victim = pred.percentile()
    np.testing.assert_equal(victim, goal)
