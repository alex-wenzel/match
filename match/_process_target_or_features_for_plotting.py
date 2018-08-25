from numpy import nanmax, nanmin
from pandas import DataFrame, Series

from .nd_array.nd_array.normalize_nd_array import normalize_nd_array
from .plot.plot.make_colorscale import make_colorscale
from .plot.plot.style import (BINARY_COLORS_WHITE_BLACK, CATEGORICAL_COLORS,
                              CONTINUOUS_COLORSCALE_FOR_MATCH)


def _process_target_or_features_for_plotting(
        target_or_features,
        type_,
        plot_std_max,
):

    if isinstance(
            target_or_features,
            Series,
    ):

        is_target = True

    elif isinstance(
            target_or_features,
            DataFrame,
    ):

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
                    None,
                    '-0-',
                    raise_for_bad_value=False,
                ),
                name=target_or_features.name,
                index=target_or_features.index,
            )

        else:

            target_or_features = DataFrame(
                normalize_nd_array(
                    target_or_features.values,
                    1,
                    '-0-',
                    raise_for_bad_value=False,
                ),
                index=target_or_features.index,
                columns=target_or_features.columns,
            )

        target_or_features_nanmin = nanmin(target_or_features.values)

        target_or_features_nanmax = nanmax(target_or_features.values)

        if plot_std_max is None:

            plot_min = target_or_features_nanmin

            plot_max = target_or_features_nanmax

        else:

            plot_min = max(
                -plot_std_max,
                target_or_features_nanmin,
            )

            plot_max = min(
                plot_std_max,
                target_or_features_nanmax,
            )

        colorscale = CONTINUOUS_COLORSCALE_FOR_MATCH

    else:

        plot_min = None

        plot_max = None

        if type_ == 'categorical':

            if is_target:

                n_color = target_or_features.unique().size

            else:

                n_color = target_or_features.unstack().unique().size

            colorscale = make_colorscale(colors=CATEGORICAL_COLORS[:n_color])

        elif type_ == 'binary':

            colorscale = make_colorscale(colors=BINARY_COLORS_WHITE_BLACK)

        else:

            raise ValueError('Unknown type_: {}.'.format(type_))

    return target_or_features, plot_min, plot_max, colorscale
