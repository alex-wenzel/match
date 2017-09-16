from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from pandas import DataFrame
from seaborn import heatmap

from .plot.plot.decorate import decorate
from .plot.plot.plot import save_plot
from .plot.plot.style import FONT_LARGEST, FONT_STANDARD
from .prepare_data_for_plotting import prepare_data_for_plotting
from .support.support.iterable import get_uniques_in_order

SPACING = 0.05


def plot_match(target,
               features,
               annotations,
               target_type,
               features_type,
               title,
               plot_sample_names,
               file_path,
               target_ax=None,
               features_ax=None):
    """
    Plot matches.
    Arguments:
        target (Series): (n_samples)
        features (DataFrame): (n_features, n_samples)

        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        title (str): Plot title
        plot_sample_names (bool): Whether to plot column names
        file_path (str):
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
    figure(figsize=(min(pow(features.shape[1], 0.7), 7), pow(
        features.shape[0], 0.9)))

    # Set up grids & axes
    if target_ax is None or features_ax is None:
        gridspec = GridSpec(features.shape[0] + 1, 1)
        target_ax = subplot(gridspec[:1, 0])
        features_ax = subplot(gridspec[1:, 0])

    #
    # Plot target, target label, & title
    #
    # Plot target
    heatmap(
        DataFrame(target).T,
        ax=target_ax,
        vmin=target_min,
        vmax=target_max,
        cmap=target_cmap,
        xticklabels=False,
        yticklabels=bool(target.name),
        cbar=False)

    # Adjust target name
    decorate(
        ax=target_ax, despine_kwargs={'left': True,
                                      'bottom': True}, ylabel='')

    if target_type in ('binary', 'categorical'):  # Add labels

        # Get boundary indices
        boundary_is = [0]
        prev_v = target[0]
        for i, v in enumerate(target[1:]):
            if prev_v != v:
                boundary_is.append(i + 1)
            prev_v = v
        boundary_is.append(features.shape[1])

        # Get positions
        label_xs = []
        prev_i = 0
        for i in boundary_is[1:]:
            label_xs.append(i - (i - prev_i) / 2)
            prev_i = i

        # Plot values to their corresponding positions
        unique_target_labels = get_uniques_in_order(target.values)
        for i, x in enumerate(label_xs):
            target_ax.text(
                x,
                target_ax.axis()[3] * (1 + SPACING),
                unique_target_labels[i],
                horizontalalignment='center',
                **FONT_STANDARD)

    if title:  # Plot title
        target_ax.text(
            target_ax.axis()[1] / 2,
            -target_ax.axis()[2] / 2,
            title,
            horizontalalignment='center',
            **FONT_LARGEST)

    # Plot annotation header
    target_ax.text(
        target_ax.axis()[1] + target_ax.axis()[1] * SPACING,
        target_ax.axis()[2] / 2,
        ' ' * 6 + 'IC(\u0394)' + ' ' * 12 + 'p-value' + ' ' * 12 + 'FDR',
        verticalalignment='center',
        **FONT_STANDARD)

    heatmap(
        features,
        ax=features_ax,
        vmin=features_min,
        vmax=features_max,
        cmap=features_cmap,
        xticklabels=plot_sample_names,
        cbar=False)

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
        save_plot(file_path)
