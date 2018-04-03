from numpy import nanmax, nanmin
from pandas import DataFrame, Series

from .nd_array.nd_array.normalize_1d_array import normalize_1d_array
from .nd_array.nd_array.normalize_2d_array import normalize_2d_array
from .plot.plot.make_colorscale import make_colorscale
from .plot.plot.style import (BLACK_WHITE_BINARY_COLORS, CATEGORICAL_COLORS,
                              COLORSCALE_FOR_MATCH)


def process_target_and_features_for_plotting(target, target_type, features,
                                             features_type, plot_max_std):

    if target_type == 'continuous':
        target = Series(
            normalize_1d_array(
                target.values, method='-0-', ignore_bad_value=True).clip(
                    -plot_max_std, plot_max_std),
            name=target.name,
            index=target.index)

        target_min = nanmin(target.values)
        target_max = nanmax(target.values)
        target_colorscale = COLORSCALE_FOR_MATCH

    else:
        target_min = 0
        if target_type == 'categorical':
            target_max = target.unique().size - 1
            target_colorscale = make_colorscale(CATEGORICAL_COLORS)
        elif target_type == 'binary':
            target_max = 1
            target_colorscale = make_colorscale(BLACK_WHITE_BINARY_COLORS)
        else:
            raise ValueError('Unknown target_type: {}.'.format(target_type))

    target = target.to_frame().T

    if features_type == 'continuous':
        features = DataFrame(
            normalize_2d_array(
                features.values, method='-0-', axis=1,
                ignore_bad_value=True).clip(-plot_max_std, plot_max_std),
            index=features.index,
            columns=features.columns)

        features_min = nanmin(features)
        features_max = nanmax(features)
        features_colorscale = COLORSCALE_FOR_MATCH

    else:
        features_min = 0
        if features_type == 'categorical':
            features_max = features.unstack().unique().size - 1
            features_colorscale = make_colorscale(CATEGORICAL_COLORS)
        elif features_type == 'binary':
            features_max = 1
            features_colorscale = make_colorscale(BLACK_WHITE_BINARY_COLORS)
        else:
            raise ValueError(
                'Unknown features_type: {}.'.format(features_type))

    return target, target_min, target_max, target_colorscale, features, features_min, features_max, features_colorscale
