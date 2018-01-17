import pytest
import pandas as pd
from deltaoxfox.modelparams import get_draws


class TestSeawaterDraws:
    def test_find_nearest_latlon(self):
        target_lat = 42
        target_lon = -120
        goal = pd.DataFrame({'lat': [40.0, 40.0], 'lon': [-130.0, -110.0]},
                            index=[0, 70])
        d18osw = get_draws('d18osw')
        victim = d18osw.find_nearest_latlon(target_lat, target_lon)
        pd.testing.assert_frame_equal(victim, goal)
