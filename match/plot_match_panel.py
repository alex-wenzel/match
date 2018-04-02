from matplotlib.colorbar import ColorbarBase, make_axes
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from numpy import nanmax, nanmean, nanmin
from pandas import DataFrame, Series
from seaborn import heatmap

from .nd_array.nd_array.normalize_1d_array import normalize_1d_array
from .nd_array.nd_array.normalize_2d_array import normalize_2d_array
from .plot.old_plot.decorate_ax import decorate_ax
from .plot.old_plot.save_plot import save_plot
from .plot.old_plot.style import (CMAP_BINARY_WB, CMAP_CATEGORICAL,
                                  CMAP_CONTINUOUS_BWR2, FIGURE_SIZE,
                                  FONT_LARGEST, FONT_STANDARD)


def plot_match_panel(target, features, target_type, features_type,
                     plot_max_std, target_ax, features_ax, title,
                     target_xticklabels, max_ytick_size, annotations,
                     plot_column_names, file_path):

    if target_type == 'continuous':
        target = Series(
            normalize_1d_array(
                target.values, method='-0-', ignore_bad_value=True).clip(
                    -plot_max_std, plot_max_std),
            name=target.name,
            index=target.index)

        target_min = nanmin(target.values)
        target_max = nanmax(target.values)
        target_cmap = CMAP_CONTINUOUS_BWR2

    else:
        target_min = 0
        target_max = target.unique().size

        if target_type == 'categorical':
            target_cmap = CMAP_CATEGORICAL

        elif target_type == 'binary':
            target_cmap = CMAP_BINARY_WB

        else:
            raise ValueError('Unknown target_type: {}.'.format(target_type))

    if features_type == 'continuous':
        features = DataFrame(
            normalize_2d_array(
                features.values, method='-0-', axis=1,
                ignore_bad_value=True).clip(-plot_max_std, plot_max_std),
            index=features.index,
            columns=features.columns)

        features_min = nanmin(features)
        features_max = nanmax(features)
        features_cmap = CMAP_CONTINUOUS_BWR2

    else:
        features_min = 0
        features_max = features.unstack().unique().size

        if features_type == 'categorical':
            features_cmap = CMAP_CATEGORICAL

        elif features_type == 'binary':
            features_cmap = CMAP_BINARY_WB

        else:
            raise ValueError(
                'Unknown features_type: {}.'.format(features_type))

    if target_ax is None or features_ax is None:
        min(pow(features.shape[1], 1.8)

    #

    title

    target_ax.text(
        target_ax.get_xlim()[1] * 1.018,
        0.5,
        '   |   '.join(annotations.columns).expandtabs(),
        verticalalignment='center',
        **FONT_STANDARD)

    #

    for i, (index, annotations_) in enumerate(annotations.iterrows()):
            '   '.join(annotations_)
