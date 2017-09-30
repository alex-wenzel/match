from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from numpy import unique
from pandas import DataFrame, Series
from seaborn import heatmap

from .array_nd.array_nd.normalize_1d_array import normalize_1d_array
from .array_nd.array_nd.normalize_2d_array import normalize_2d_array
from .plot.plot.decorate import decorate
from .plot.plot.make_random_colormap import make_random_colormap
from .plot.plot.save_plot import save_plot
from .plot.plot.style import (CMAP_BINARY_BW, CMAP_CATEGORICAL_TAB20,
                              CMAP_CONTINUOUS_ASSOCIATION, FIGURE_SIZE,
                              FONT_LARGEST, FONT_STANDARD)
from .support.support.dict_ import merge_dicts_with_function
from .support.support.iterable import get_unique_objects_in_order


def plot_match_panel(target, target_int_to_o, features, max_std, annotations,
                     figure_size, target_ax, features_ax, target_type,
                     features_type, title, target_annotation_kwargs,
                     plot_sample_names, file_path, dpi):
    """
    Plot matches.
    Arguments:
        target (Series): (n_samples)
        target_int_to_o (dict):
        features (DataFrame): (n_features, n_samples)
        max_std (number):
        annotations (DataFrame): (n_features, 3)
        figure_size (tuple):
        target_ax (matplotlib ax):
        features_ax (matplotlib ax):
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        title (str): plot title
        target_annotation_kwargs (dict):
        plot_sample_names (bool): whether to plot column names
        file_path (str):
        dpi (int):
    Returns:
        None
    """

    # Set target min, max, and colormap
    if target_type == 'continuous':
        # Normalize target for plotting
        target = Series(
            normalize_1d_array(target.values, method='-0-'),
            name=target.name,
            index=target.index)
        target_min, target_max, target_cmap = -max_std, max_std, CMAP_CONTINUOUS_ASSOCIATION

    elif target_type == 'categorical':
        n = unique(target).size
        if CMAP_CATEGORICAL_TAB20.N < n:
            # Make and use a Colormap with random colors
            cmap = make_random_colormap(n_colors=n)
        else:
            cmap = CMAP_CATEGORICAL_TAB20
        target_min, target_max, target_cmap = 0, n, cmap

    elif target_type == 'binary':
        target_min, target_max, target_cmap = 0, 1, CMAP_BINARY_BW

    else:
        raise ValueError('Unknown target_type: {}.'.format(target_type))

    # Set features min, max, and colormap
    if features_type == 'continuous':
        # Normalize featuers for plotting
        features = DataFrame(
            normalize_2d_array(features.values, method='-0-', axis=1),
            index=features.index,
            columns=features.columns)
        features_min, features_max, features_cmap = -max_std, max_std, CMAP_CONTINUOUS_ASSOCIATION

    elif features_type == 'categorical':
        n = unique(features).size
        if CMAP_CATEGORICAL_TAB20.N < n:
            # Make and use a Colormap with random colors
            cmap = make_random_colormap(n_colors=n)
        else:
            cmap = CMAP_CATEGORICAL_TAB20
        features_min, features_max, features_cmap = 0, n, cmap

    elif features_type == 'binary':
        features_min, features_max, features_cmap = 0, 1, CMAP_BINARY_BW

    else:
        raise ValueError('Unknown features_type: {}.'.format(features_type))

    # Set up figure
    if not figure_size:
        figure_size = (min(pow(features.shape[1], 1.8), FIGURE_SIZE[1]),
                       features.shape[0])

    # Set up grids and axes if target_ax or features_ax is not specified
    if target_ax is None or features_ax is None:
        figure(figsize=figure_size)
        gridspec = GridSpec(features.shape[0] + 1, 1)
        target_ax = subplot(gridspec[:1, 0])
        features_ax = subplot(gridspec[1:, 0])

    # Plot target heatmap
    heatmap(
        DataFrame(target).T,
        ax=target_ax,
        vmin=target_min,
        vmax=target_max,
        cmap=target_cmap,
        xticklabels=False,
        yticklabels=bool(target.name),
        cbar=False)

    # Decorate target heatmap
    decorate(
        ax=target_ax, despine_kwargs={'left': True,
                                      'bottom': True}, ylabel='')

    # Plot title
    if title:

        target_ax.text(
            target_ax.get_xlim()[1] / 2,
            -1,
            title,
            horizontalalignment='center',
            **merge_dicts_with_function(FONT_LARGEST, {'color': '#9017E6'},
                                        lambda a, b: b))

    # Plot target label
    if target_type in ('binary', 'categorical'):

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
        unique_target_labels = get_unique_objects_in_order(target.values)
        for i, x in enumerate(label_positions):

            if target_int_to_o:
                t = target_int_to_o[unique_target_labels[i]]

            target_ax.text(
                x,
                -0.18,
                t,
                horizontalalignment='center',
                verticalalignment='bottom',
                rotation=90,
                **merge_dicts_with_function(
                    FONT_STANDARD, target_annotation_kwargs, lambda a, b: b))

    # Plot annotation header
    target_ax.text(
        target_ax.get_xlim()[1] * 1.018,
        0.5,
        ' ' * 3 + 'IC(\u0394)' + ' ' * 15 + 'p-value' + ' ' * 12 + 'FDR',
        verticalalignment='center',
        **FONT_STANDARD)

    # Plot annotation header separator line
    target_ax.plot(
        [target_ax.get_xlim()[1] * 1.02,
         target_ax.get_xlim()[1] * 1.4], [1, 1],
        '-',
        linewidth=1,
        color='#20D9BA',
        clip_on=False,
        aa=True)

    # Plot features heatmap
    heatmap(
        features,
        ax=features_ax,
        vmin=features_min,
        vmax=features_max,
        cmap=features_cmap,
        xticklabels=plot_sample_names,
        cbar=False)

    # Decorate features heatmap
    decorate(
        ax=features_ax,
        despine_kwargs={
            'left': True,
            'bottom': True,
        },
        ylabel='')

    # Plot annotations
    for i, (a_i, a) in enumerate(annotations.iterrows()):
        features_ax.text(
            target_ax.axis()[1] * 1.018,
            i + 0.5,
            '\t'.join(a.tolist()).expandtabs(),
            verticalalignment='center',
            **FONT_STANDARD)

    # Save
    if file_path:
        save_plot(file_path, dpi=dpi)
