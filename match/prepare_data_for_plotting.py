from numpy import unique
from pandas import DataFrame, Series

from .array_nd.array_nd.array_1d import normalize as array_1d_normalize
from .array_nd.array_nd.array_2d import normalize as array_2d_normalize
from .plot.plot.style import (CMAP_BINARY_BW, CMAP_CATEGORICAL_TAB20,
                              CMAP_CONTINUOUS_ASSOCIATION)


def _prepare_data_for_plotting(a, data_type, max_std=3):
    """
    Normalize a and return good min, max, and matplotlib.cm for plotting.
    Arguments:
         a (array): (n) | (n, m)
         data_type (str): 'continuous' | 'categorical' | 'binary'
         max_std (number):
    Returns:
         Series | DataFrame:
         float: Minimum
         float: Maximum
         matplotlib.cm:
    """

    if data_type == 'continuous':

        if a.ndim == 1:
            a_ = array_1d_normalize(a.values, method='-0-')
            a = Series(a_, name=a.name, index=a.index)

        elif a.ndim == 2:
            a_ = array_2d_normalize(a.values, method='-0-', axis=1)
            a = DataFrame(a_, index=a.index, columns=a.columns)

        return a, -max_std, max_std, CMAP_CONTINUOUS_ASSOCIATION

    elif data_type == 'categorical':
        return a.copy(), 0, unique(a).size, CMAP_CATEGORICAL_TAB20

    elif data_type == 'binary':
        return a.copy(), 0, 1, CMAP_BINARY_BW
