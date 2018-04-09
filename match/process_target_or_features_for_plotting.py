from numpy import nanmax, nanmin
from pandas import DataFrame, Series

from .nd_array.nd_array.normalize_1d_array import normalize_1d_array
from .nd_array.nd_array.normalize_2d_array import normalize_2d_array
from .plot.plot.make_colorscale import make_colorscale
from .plot.plot.style import (BINARY_COLORS_WHITE_BLACK, CATEGORICAL_COLORS,
                              CONTINUOUS_COLORSCALE_FOR_MATCH)


def process_target_or_features_for_plotting(target_or_features, type_,
                                            plot_max_std):

    if isinstance(target_or_features, Series):
        is_target = True
    elif isinstance(target_or_features, DataFrame):
        is_target = False
    else:
        raise ValueError(
            'target_or_features ({}) is neither a Series or DataFrame.'.format(
                type(target_or_features)))

    if type_ == 'continuous':

        if is_target:
            target_or_features = Series(
                normalize_1d_array(
                    target_or_features.values,
                    method='-0-',
                    ignore_bad_value=True),
                name=target_or_features.name,
                index=target_or_features.index)

        else:
            target_or_features = DataFrame(
                normalize_2d_array(
                    target_or_features.values,
                    method='-0-',
                    axis=1,
                    ignore_bad_value=True),
                index=target_or_features.index,
                columns=target_or_features.columns)

        target_or_features.clip(-plot_max_std, plot_max_std),

        min_ = nanmin(target_or_features)
        max_ = nanmax(target_or_features)
        colorscale = CONTINUOUS_COLORSCALE_FOR_MATCH

    else:
        min_ = 0

        if type_ == 'categorical':
            if is_target:
                max_ = target_or_features.unique().size - 1
            else:
                max_ = target_or_features.unstack().unique().size - 1
            colorscale = make_colorscale(CATEGORICAL_COLORS)

        elif type_ == 'binary':
            max_ = 1
            colorscale = make_colorscale(BINARY_COLORS_WHITE_BLACK)

        else:
            raise ValueError('Unknown type_: {}.'.format(type_))

    return target_or_features, min_, max_, colorscale
