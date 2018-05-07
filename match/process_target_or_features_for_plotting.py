from numpy import nanmax, nanmin
from pandas import DataFrame, Series

from .nd_array.nd_array.normalize_nd_array import normalize_nd_array
from .plot.plot.make_colorscale import make_colorscale
from .plot.plot.style import (BINARY_COLORS_WHITE_BLACK, CATEGORICAL_COLORS,
                              CONTINUOUS_COLORSCALE_FOR_MATCH)


def process_target_or_features_for_plotting(target_or_features, type_,
                                            plot_std_max):

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
                normalize_nd_array(
                    target_or_features.values,
                    '-0-',
                    None,
                    raise_for_bad_value=False),
                name=target_or_features.name,
                index=target_or_features.index)

        else:

            target_or_features = DataFrame(
                normalize_nd_array(
                    target_or_features.values,
                    '-0-',
                    1,
                    raise_for_bad_value=False),
                index=target_or_features.index,
                columns=target_or_features.columns)

        min_ = max(-plot_std_max, nanmin(target_or_features.values))

        max_ = min(plot_std_max, nanmax(target_or_features.values))

        colorscale = CONTINUOUS_COLORSCALE_FOR_MATCH

    else:

        min_ = 0

        if type_ == 'categorical':

            if is_target:

                max_ = target_or_features.unique().size - 1

            else:

                max_ = target_or_features.unstack().unique().size - 1

            colorscale = make_colorscale(colors=CATEGORICAL_COLORS)

        elif type_ == 'binary':

            max_ = 1

            colorscale = make_colorscale(colors=BINARY_COLORS_WHITE_BLACK)

        else:

            raise ValueError('Unknown type_: {}.'.format(type_))

    return target_or_features, min_, max_, colorscale
