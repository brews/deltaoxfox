import deltaoxfox as dfox
import numpy as np


def test_foram_sst_minmax():
    """Simple integration test for ``foram_sst_minmax``"""
    vmin, vmax = dfox.foram_sst_minmax('G. ruber white')
    np.testing.assert_allclose(vmin, 10.9, atol=1e-1)
    np.testing.assert_allclose(vmax, 29.5, atol=1e-1)
