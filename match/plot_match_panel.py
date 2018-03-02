from matplotlib.colorbar import ColorbarBase, make_axes
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from numpy import nanmax, nanmean, nanmin
from pandas import DataFrame, Series
from seaborn import heatmap

from .nd_array.nd_array.normalize_1d_array import normalize_1d_array
from .nd_array.nd_array.normalize_2d_array import normalize_2d_array
from .plot.plot.decorate_ax import decorate_ax
from .plot.plot.save_plot import save_plot
from .plot.plot.style import (CMAP_BINARY_WB, CMAP_CATEGORICAL,
                              CMAP_CONTINUOUS_BWR2, FIGURE_SIZE, FONT_LARGEST,
                              FONT_STANDARD)


def plot_match_panel(target, features, target_type, features_type, target_ax,
                     features_ax, title, target_xticklabels, max_ytick_size,
                     annotations, plot_column_names, file_path):
    """
    Plot match panel.
    Arguments:
        target (Series): (n_sample, )
        features (DataFrame): (n_feature, n_sample, )
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        target_ax (matplotlib.Axes):
        features_ax (matplotlib.Axes):
        title (str):
        target_xticklabels (iterable):
        max_ytick_size (int):
        annotations (DataFrame): (n_feature, 3, )
        plot_column_names (bool):
        file_path (str):
    Returns:
    """

    max_std = 3

    if target_type == 'continuous':
        target = Series(
            normalize_1d_array(target.values, method='-0-').clip(
                -max_std, max_std),
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
            normalize_2d_array(features.values, method='-0-', axis=1).clip(
                -max_std, max_std),
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
        figure(figsize=(min(pow(features.shape[1], 1.8), FIGURE_SIZE[1]),
                        features.shape[0]))

        gridspec = GridSpec(features.shape[0] + 1, 1)
        target_ax = subplot(gridspec[:1, 0])
        features_ax = subplot(gridspec[1:-1, 0])
        colorbar_ax = subplot(gridspec[-1:, 0])
        colorbar_ax.set_axis_off()

        if features_type == 'continuous':
            colorbar_ax_, kwargs = make_axes(
                colorbar_ax,
                location='bottom',
                fraction=0.2,
                cmap=features_cmap,
                norm=Normalize(vmin=features_min, vmax=features_max),
                ticks=(
                    features_min,
                    nanmean(features),
                    features_max, ))
            ColorbarBase(colorbar_ax_, **kwargs)

            decorate_ax(colorbar_ax_)

    if len(target_xticklabels) and len(target_xticklabels) != target.size:
        raise ValueError(
            'The sizes of target_xticklabels and target mismatch.')
    heatmap(
        DataFrame(target).T,
        ax=target_ax,
        cmap=target_cmap,
        vmin=target_min,
        vmax=target_max,
        xticklabels=target_xticklabels,
        yticklabels=(target.name, ),
        cbar=False)

    decorate_ax(
        target_ax,
        despine_kwargs={
            'left': True,
            'bottom': True,
        },
        xaxis_position='top',
        max_ytick_size=max_ytick_size)

    if title:
        target_ax.text(
            target_ax.get_xlim()[1] / 2,
            -1,
            title,
            horizontalalignment='center',
            **FONT_LARGEST)

    target_ax.text(
        target_ax.get_xlim()[1] * 1.018,
        0.5,
        ' ' * 5 + 'IC(\u0394)' + ' ' * 12 + 'P-Value' + ' ' * 11 + 'FDR',
        verticalalignment='center',
        **FONT_STANDARD)

    heatmap(
        features,
        ax=features_ax,
        cmap=features_cmap,
        vmin=features_min,
        vmax=features_max,
        xticklabels=plot_column_names,
        cbar=False)

    decorate_ax(
        features_ax,
        despine_kwargs={
            'left': True,
            'bottom': True,
        },
        xlabel='',
        ylabel='',
        max_ytick_size=max_ytick_size)

    for i, (
            a_i,
            a, ) in enumerate(annotations.iterrows()):

        features_ax.text(
            target_ax.axis()[1] * 1.018,
            i + 0.5,
            '\t'.join(a.tolist()).expandtabs(),
            verticalalignment='center',
            **FONT_STANDARD)

    if file_path:
        save_plot(file_path)
