from matplotlib.colorbar import ColorbarBase, make_axes
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from pandas import DataFrame, isna
from seaborn import heatmap

from .plot.plot.decorate_ax import decorate_ax
from .plot.plot.save_plot import save_plot
from .plot.plot.style import (CMAP_BINARY_WB, CMAP_CATEGORICAL,
                              CMAP_CONTINUOUS_ASSOCIATION, FIGURE_SIZE,
                              FONT_LARGEST, FONT_STANDARD)


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

    if target_type == 'continuous':
        target_cmap = CMAP_CONTINUOUS_ASSOCIATION
    elif target_type == 'categorical':
        target_cmap = CMAP_CATEGORICAL
    elif target_type == 'binary':
        target_cmap = CMAP_BINARY_WB
    else:
        raise ValueError('Unknown target_type: {}.'.format(target_type))

    if features_type == 'continuous':
        features_cmap = CMAP_CONTINUOUS_ASSOCIATION
    elif features_type == 'categorical':
        features_cmap = CMAP_CATEGORICAL
    elif features_type == 'binary':
        features_cmap = CMAP_BINARY_WB
    else:
        raise ValueError('Unknown features_type: {}.'.format(features_type))

    if target_ax is None or features_ax is None:

        figure(figsize=(min(pow(features.shape[1], 1.8), FIGURE_SIZE[1]),
                        features.shape[0]))

        gridspec = GridSpec(features.shape[0] + 1, 1)
        target_ax = subplot(gridspec[:1, 0])
        features_ax = subplot(gridspec[1:-1, 0])
        colorbar_ax = subplot(gridspec[-1:, 0])
        colorbar_ax.axis('off')

        features_values = features.values[~isna(features)]
        if features_type == 'continuous':
            cax, kw = make_axes(
                colorbar_ax,
                location='bottom',
                fraction=0.2,
                cmap=features_cmap,
                ticks=(
                    features_values.min(),
                    features_values.mean(),
                    features_values.max(), ))
            ColorbarBase(cax, **kw)
            decorate_ax(cax)

        save_plot_ = True

    else:
        save_plot_ = False

    if len(target_xticklabels) != target.size:
        raise ValueError(
            'The sizes of target_xticklabels and target mismatch.')
    heatmap(
        DataFrame(target).T,
        ax=target_ax,
        cmap=target_cmap,
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

    if save_plot_:

        if file_path:
            save_plot(file_path)
