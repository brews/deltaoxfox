import pytest
import numpy as np
import pandas as pd
from deltaoxfox.modelparams import get_draws
from deltaoxfox.modelparams.core import chord_distance


@pytest.mark.parametrize("test_input,expected", [
    ([(17.3, -48.4), (17.5, -48.5)], 24.668176282908473),
    ([(17.3, -48.4), [(17.5, -48.5), (-69.5, -179.5)]],
     np.array([[24.668176282908473], [1.104116337324761e+04]])),
])
def test_chord_distance(test_input, expected):
    """Test chord_distance against inputs of multiple size"""
    victim = chord_distance(*test_input)
    np.testing.assert_allclose(victim, expected, atol=1e-8)


class TestSeawaterDraws:
    @pytest.mark.parametrize("test_input,expected", [
        ((42, -120), pd.DataFrame({'lat': [40.0, 40.0], 'lon': [-130.0, -110.0]}, index=[0, 70])),
        ((-4.816667, 39.423667), pd.DataFrame({'lat': [0.0], 'lon': [50.0]}, index=[124])),
    ])
    def test_find_nearest_latlon(self, test_input, expected):
        d18osw = get_draws('d18osw')
        victim = d18osw.find_nearest_latlon(*test_input)
        pd.testing.assert_frame_equal(victim, expected)
