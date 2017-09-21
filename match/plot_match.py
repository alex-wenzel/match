from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from pandas import DataFrame
from seaborn import heatmap

from .plot.plot.decorate import decorate
from .plot.plot.plot import save_plot
from .plot.plot.style import FONT_LARGEST, FONT_SMALLEST, FONT_STANDARD
from .prepare_data_for_plotting import prepare_data_for_plotting
from .support.support.iterable import get_uniques_in_order

SPACING = 0.05


def plot_match(target,
               target_int_to_o,
               features,
               annotations,
               figure_size,
               target_type,
               features_type,
               title,
               plot_sample_names,
               file_path,
               dpi,
               target_ax=None,
               features_ax=None):
    """
    Plot matches.
    Arguments:
        target (Series): (n_samples)
        target_int_to_o (dict):
        features (DataFrame): (n_features, n_samples)
        annotations (DataFrame): (n_features, 3)
        figure_size (tuple):
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        title (str): Plot title
        plot_sample_names (bool): Whether to plot column names
        file_path (str):
        dpi (int):
        target_ax (matplotlib ax):
        features_ax (matplotlib ax):
    Returns:
        None
    """

    # Prepare target for plotting
    target, target_min, target_max, target_cmap = prepare_data_for_plotting(
        target, target_type)

    # Prepare features for plotting
    features, features_min, features_max, features_cmap = prepare_data_for_plotting(
        features, features_type)

    # Set up figure
    if not figure_size:
        figure_size = (min(pow(features.shape[1], 0.8), 10), pow(
            features.shape[0], 0.8))

    figure(figsize=figure_size)

    # Set up grids & axes
    if target_ax is None or features_ax is None:
        gridspec = GridSpec(features.shape[0] + 1, 1)
        target_ax = subplot(gridspec[:1, 0])
        features_ax = subplot(gridspec[1:, 0])

    # Plot title, target heatmap, target label, and annotation header
    if target_ax:

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
            ax=target_ax,
            despine_kwargs={'left': True,
                            'bottom': True},
            ylabel='')

        # Plot title
        if title:

            target_ax.text(
                target_ax.axis()[1] / 2,
                -target_ax.axis()[2],
                title,
                horizontalalignment='center',
                **FONT_LARGEST)

        # Plot target label
        if target_type in ('binary', 'categorical'):

            # Get boundary index
            boundary_indexs = [0]
            prev_v = target[0]
            for i, v in enumerate(target[1:]):
                if prev_v != v:
                    boundary_indexs.append(i + 1)
                prev_v = v
            boundary_indexs.append(features.shape[1])

            # Get label position
            label_positions = []
            prev_i = 0
            for i in boundary_indexs[1:]:
                label_positions.append(i - (i - prev_i) / 2)
                prev_i = i

            # Plot target label
            unique_target_labels = get_uniques_in_order(target.values)
            for i, x in enumerate(label_positions):

                if target_int_to_o:
                    t = target_int_to_o[unique_target_labels[i]]

                target_ax.text(
                    x,
                    -target_ax.axis()[2] / 8,
                    t,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    rotation=90,
                    **FONT_SMALLEST)

        # Plot annotation header
        target_ax.text(
            target_ax.axis()[1] + target_ax.axis()[1] * SPACING,
            target_ax.axis()[2] / 2,
            ' ' * 6 + 'IC(\u0394)' + ' ' * 12 + 'p-value' + ' ' * 12 + 'FDR',
            verticalalignment='center',
            **FONT_STANDARD)

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
        despine_kwargs={'left': True,
                        'bottom': True},
        ylabel='')

    # Plot annotations
    for i, (a_i, a) in enumerate(annotations.iterrows()):
        features_ax.text(
            features_ax.axis()[1] + features_ax.axis()[1] * SPACING,
            features_ax.axis()[3] + i + 0.5,
            '\t'.join(a.tolist()).expandtabs(),
            verticalalignment='center',
            **FONT_STANDARD)

    # Save
    if file_path:
        save_plot(file_path, dpi=dpi)
