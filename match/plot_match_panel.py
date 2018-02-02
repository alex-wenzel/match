from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from numpy import unique
from pandas import DataFrame, Series
from seaborn import heatmap

from .nd_array.nd_array.normalize_1d_array import normalize_1d_array
from .nd_array.nd_array.normalize_2d_array import normalize_2d_array
from .plot.plot.decorate_ax import decorate_ax
from .plot.plot.make_categorical_colormap import make_categorical_colormap
from .plot.plot.save_plot import save_plot
from .plot.plot.style import (CMAP_BINARY_WB, CMAP_CATEGORICAL,
                              CMAP_CONTINUOUS_ASSOCIATION, FIGURE_SIZE,
                              FONT_LARGEST, FONT_STANDARD)
from .support.support.iterable import get_unique_iterable_objects_in_order


def plot_match_panel(target, features, target_type, features_type, max_std,
                     target_ax, features_ax, title, target_int_to_str,
                     target_annotation_kwargs, max_ytick_size, annotations,
                     plot_column_names, file_path):
    """
    Plot match panel.
    Arguments:
        target (Series): (n_sample, )
        features (DataFrame): (n_feature, n_sample, )
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        max_std (float):
        target_ax (matplotlib.Axes):
        features_ax (matplotlib.Axes):
        title (str):
        target_int_to_str (dict):
        target_annotation_kwargs (dict):
        max_ytick_size (int):
        annotations (DataFrame): (n_feature, 3, )
        plot_column_names (bool):
        file_path (str):
    Returns:
    """

    # Set target min, max, and colormap
    if target_type == 'continuous':
        # Normalize target for plotting
        target = Series(
            normalize_1d_array(target.values, method='-0-').clip(
                -max_std, max_std),
            name=target.name,
            index=target.index)
        target_min, target_max, target_cmap = -max_std, max_std, CMAP_CONTINUOUS_ASSOCIATION

    elif target_type == 'categorical':
        n = unique(target).size
        if CMAP_CATEGORICAL.N < n:
            cmap = make_categorical_colormap()
        else:
            cmap = CMAP_CATEGORICAL
        target_min, target_max, target_cmap = 0, n, cmap

    elif target_type == 'binary':
        target_min, target_max, target_cmap = 0, 1, CMAP_BINARY_WB

    else:
        raise ValueError('Unknown target_type: {}.'.format(target_type))

    # Set features min, max, and colormap
    if features_type == 'continuous':
        # Normalize features for plotting
        features = DataFrame(
            normalize_2d_array(features.values, method='-0-', axis=1).clip(
                -max_std, max_std),
            index=features.index,
            columns=features.columns)
        features_min, features_max, features_cmap = -max_std, max_std, CMAP_CONTINUOUS_ASSOCIATION

    elif features_type == 'categorical':
        n = unique(features).size
        if CMAP_CATEGORICAL.N < n:
            cmap = make_categorical_colormap()
        else:
            cmap = CMAP_CATEGORICAL
        features_min, features_max, features_cmap = 0, n, cmap

    elif features_type == 'binary':
        features_min, features_max, features_cmap = 0, 1, CMAP_BINARY_WB

    else:
        raise ValueError('Unknown features_type: {}.'.format(features_type))

    if target_ax is None or features_ax is None:
        # Set up figure and grids and axes

        figure(figsize=(min(pow(features.shape[1], 1.8), FIGURE_SIZE[1]),
                        features.shape[0]))

        gridspec = GridSpec(features.shape[0] + 1, 1)

        target_ax = subplot(gridspec[:1, 0])
        features_ax = subplot(gridspec[1:, 0])

        save_and_show_plot = True

    else:
        save_and_show_plot = False

    # Plot target heatmap
    heatmap(
        DataFrame(target).T,
        ax=target_ax,
        vmin=target_min,
        vmax=target_max,
        cmap=target_cmap,
        xticklabels=False,
        yticklabels=[target.name],
        cbar=False)

    # Decorate target heatmap
    decorate_ax(
        target_ax,
        despine_kwargs={
            'left': True,
            'bottom': True,
        },
        xlabel='',
        ylabel='',
        max_ytick_size=max_ytick_size)

    # Plot title
    if title:
        target_ax.text(
            target_ax.get_xlim()[1] / 2,
            -1,
            title,
            horizontalalignment='center',
            **FONT_LARGEST)

    # Plot target label
    if target_type in (
            'binary',
            'categorical', ):

        # Get boundary index
        boundary_indices = [0]
        prev_v = target[0]
        for i, v in enumerate(target[1:]):
            if prev_v != v:
                boundary_indices.append(i + 1)
            prev_v = v
        boundary_indices.append(features.shape[1])

        # Get label position
        label_positions = []
        prev_i = 0
        for i in boundary_indices[1:]:
            label_positions.append(i - (i - prev_i) / 2)
            prev_i = i

        # Plot target label
        unique_target_labels = get_unique_iterable_objects_in_order(
            target.values)
        for i, x in enumerate(label_positions):

            if target_int_to_str:
                t = target_int_to_str[unique_target_labels[i]]
            else:
                t = unique_target_labels[i]

            target_ax.text(
                x,
                -0.18,
                t,
                horizontalalignment='center',
                verticalalignment='bottom',
                rotation=90,
                **{
                    **FONT_STANDARD,
                    **target_annotation_kwargs,
                })

    # Plot annotation header
    target_ax.text(
        target_ax.get_xlim()[1] * 1.018,
        0.5,
        ' ' * 5 + 'IC(\u0394)' + ' ' * 12 + 'P-Value' + ' ' * 11 + 'FDR',
        verticalalignment='center',
        **FONT_STANDARD)

    # Plot features heatmap
    heatmap(
        features,
        ax=features_ax,
        vmin=features_min,
        vmax=features_max,
        cmap=features_cmap,
        xticklabels=plot_column_names,
        cbar=False)

    # Decorate features heatmap
    decorate_ax(
        features_ax,
        despine_kwargs={
            'left': True,
            'bottom': True,
        },
        xlabel='',
        ylabel='',
        max_ytick_size=max_ytick_size)

    # Plot annotations
    for i, (
            a_i,
            a, ) in enumerate(annotations.iterrows()):

        features_ax.text(
            target_ax.axis()[1] * 1.018,
            i + 0.5,
            '\t'.join(a.tolist()).expandtabs(),
            verticalalignment='center',
            **FONT_STANDARD)

    if save_and_show_plot:

        if file_path:
            save_plot(file_path)
