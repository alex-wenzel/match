from matplotlib.colorbar import ColorbarBase, make_axes
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from numpy import array, unique
from pandas import DataFrame, Series, read_table
from seaborn import heatmap

from .array_nd.array_nd.array_1d import normalize as array_1d_normalize
from .array_nd.array_nd.array_2d import normalize as array_2d_normalize
from .match import match
from .plot.plot.decorate import decorate
from .plot.plot.plot import save_plot
from .plot.plot.style import (CMAP_BINARY_BW, CMAP_CATEGORICAL_TAB20,
                              CMAP_CONTINUOUS_ASSOCIATION, FIGURE_SIZE,
                              FONT_LARGER, FONT_LARGEST, FONT_STANDARD)
from .support.support.df import drop_slices, get_top_and_bottom_indices
from .support.support.iterable import get_uniques_in_order
from .support.support.path import establish_path

RANDOM_SEED = 20121020

SPACING = 0.05


def make_match_panel(target,
                     features,
                     keep_only_target_columns_with_value=True,
                     target_ascending=False,
                     max_n_unique_objects_for_drop_slices=1,
                     result_in_ascending_order=False,
                     n_jobs=1,
                     n_features=0.99,
                     max_n_features=100,
                     n_samplings=30,
                     n_permutations=30,
                     random_seed=RANDOM_SEED,
                     target_type='continuous',
                     features_type='continuous',
                     title=None,
                     plot_sample_names=False,
                     file_path_prefix=None):
    """
    Make match panel.
        Compute: scores[i] = function(target, features[i]); confidence
        intervals (CI) for n_features features; p-values; FDRs; and plot
        n_features features.
    :param target: Series; (n_samples); DataFrame must have columns matching
        features' columns
    :param features: DataFrame; (n_features, n_samples);
    :param keep_only_target_columns_with_value: bool
    :param target_ascending: bool; True if target increase from left to right,
        and False right to left
    :param max_n_unique_objects_for_drop_slices: int
    :param result_in_ascending_order: bool; True if result increase from top to
        bottom, and False bottom to top
    :param n_jobs: int; number of multiprocess jobs
    :param n_features: number | None; number of features to compute CI and
        plot; number threshold if 1 <=, percentile threshold if < 1, and don't
        compute if None
    :param max_n_features: int;
    :param n_samplings: int; number of bootstrap samplings to build
        distribution to get CI; must be 2 < to compute CI
    :param n_permutations: int; number of permutations for permutation test to
        compute p-values and FDR
    :param random_seed: int | array;
    :param target_type: str; 'continuous' | 'categorical' | 'binary'
    :param features_type: str; 'continuous' | 'categorical' | 'binary'
    :param title: str; plot title
    :param plot_sample_names: bool; plot column names or not
    :param file_path_prefix: str; file_path_prefix.match.txt and
        file_path_prefix.match.pdf will be saved
    :return: DataFrame; (n_features, 4 ('Score', '<confidence_interval> CI',
        'p-value', 'FDR'))
    """

    target, features = _preprocess_target_and_features(
        target, features, keep_only_target_columns_with_value,
        target_ascending, max_n_unique_objects_for_drop_slices)

    scores = match(
        array(target),
        array(features),
        n_jobs=n_jobs,
        n_features=n_features,
        n_samplings=n_samplings,
        n_permutations=n_permutations,
        random_seed=random_seed)
    scores.index = features.index
    scores.sort_values(
        'Score', ascending=result_in_ascending_order, inplace=True)

    if file_path_prefix:
        file_path_txt = file_path_prefix + '.match.txt'
        file_path_pdf = file_path_prefix + '.match.pdf'
        establish_path(file_path_txt)
        scores.to_csv(file_path_txt, sep='\t')
    else:
        file_path_pdf = None

    # Keep only scores and features to plot
    indices = get_top_and_bottom_indices(
        scores, 'Score', n_features, max_n=max_n_features)

    scores_to_plot = scores.ix[indices]
    features_to_plot = features.ix[indices]

    print('Making annotations ...')
    annotations = DataFrame(index=scores_to_plot.index)
    # Add IC(confidence interval), p-value, and FDR
    annotations['IC(\u0394)'] = scores_to_plot[['Score', '0.95 CI']].apply(
        lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)
    annotations['p-value'] = scores_to_plot['p-value'].apply('{:.2e}'.format)
    annotations['FDR'] = scores_to_plot['FDR'].apply('{:.2e}'.format)

    print('Plotting match panel ...')
    _plot_match(
        target,
        features_to_plot,
        annotations,
        target_type=target_type,
        features_type=features_type,
        title=title,
        plot_sample_names=plot_sample_names,
        file_path=file_path_pdf)

    return scores


def _preprocess_target_and_features(
        target, features, keep_only_target_columns_with_value,
        target_ascending, max_n_unique_objects_for_drop_slices):
    """
    Make target Series. Select columns. Drop rows with less than
        max_n_unique_objects unique values.
    :param target: iterable | Series
    :param features: DataFrame
    :param keep_only_target_columns_with_value: bool
    :param target_ascending: bool
    :param max_n_unique_objects_for_drop_slices: int
    :return: Series & DataFrame
    """

    # Make target Series
    if not isinstance(target, Series):
        target = Series(target, index=features.columns)

    # Select columns
    if keep_only_target_columns_with_value:
        i = target.index & features.columns
        print('Target {} {} and features {} have {} shared columns.'.format(
            target.name, target.shape, features.shape, len(i)))
    else:
        i = target.index

    if not len(i):
        raise ValueError('0 column.')

    target = target[i]
    target.sort_values(ascending=target_ascending, inplace=True)

    features = features.loc[:, target.index]

    # Drop rows with less than max_n_unique_objects unique values
    features = drop_slices(
        features,
        max_n_unique_objects=max_n_unique_objects_for_drop_slices,
        axis=1)

    if features.empty:
        raise ValueError('No feature has at least {} unique objects.'.format(
            max_n_unique_objects_for_drop_slices))

    return target, features


def _plot_match(target,
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
    :param target: Series; (n_elements); must have index matching features'
        columns
    :param features: DataFrame; (n_features, n_elements)
    :param annotations: DataFrame; (n_features, n_annotations); must have index
        matching features' index
    :param target_type: str; 'continuous' | 'categorical' | 'binary'
    :param features_type: str; 'continuous' | 'categorical' | 'binary'
    :param title: str
    :param plot_sample_names: bool; plot column names or not
    :param file_path: str
    :return: None
    """

    # Prepare target for plotting
    target, target_min, target_max, target_cmap = _prepare_data_for_plotting(
        target, target_type)

    # Prepare features for plotting
    features, features_min, features_max, features_cmap = _prepare_data_for_plotting(
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
    decorate(ax=target_ax)

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

    decorate(ax=features_ax, ylabel='')

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


def make_summary_match_panel(target,
                             multiple_features,
                             keep_only_target_columns_with_value=True,
                             target_ascending=False,
                             max_n_unique_objects_for_drop_slices=1,
                             result_in_ascending_order=False,
                             target_type='continuous',
                             features_type='continuous',
                             title=None,
                             plot_sample_names=False,
                             file_path=None):
    """
    """

    # Set up figure
    fig = figure(figsize=FIGURE_SIZE)

    # Compute the number of row-grids for setting up a figure
    n = 0
    for name, features, emphasis, features_type, scores, index, alias in multiple_features:
        n += len(index) + 3

    # Add a row for color bar
    n += 1

    # Set up axis grids
    gridspec = GridSpec(n, 1)

    #
    # Annotate target with features
    #
    r_i = 0
    if not title:
        title = 'Summary Match Panel for {}'.format(title(target.name))
    fig.suptitle(title, horizontalalignment='center', **FONT_LARGEST)

    for name, features, emphasis, features_type, scores, index, alias in multiple_features:

        target, features = _preprocess_target_and_features(
            target, features, keep_only_target_columns_with_value,
            target_ascending, max_n_unique_objects_for_drop_slices)

        # Prepare target for plotting
        target, target_min, target_max, target_cmap = _prepare_data_for_plotting(
            target, target_type)

        # Prepare features for plotting
        features, features_min, features_max, features_cmap = _prepare_data_for_plotting(
            features, features_type)

        # Read corresponding match score file
        scores = read_table(scores, index_col=0)

        # Keep only selected features
        scores = scores.loc[index]

        # Sort by match score
        scores.sort_values(
            'Score', ascending=result_in_ascending_order, inplace=True)

        # Apply the sorted index to featuers
        features = features.loc[scores.index]

        i_to_a = {i: a for i, a in zip(index, alias)}
        features.index = features.index.map(lambda i: i_to_a[i])

        print('Making annotations ...')
        annotations = DataFrame(index=scores.index)
        # Add IC(confidence interval), p-value, and FDR
        annotations['IC(\u0394)'] = scores[['Score', '0.95 CI']].apply(
            lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)
        annotations['p-value'] = scores['p-value'].apply('{:.2e}'.format)
        annotations['FDR'] = scores['FDR'].apply('{:.2e}'.format)

        # Set up axes
        r_i += 1
        title_ax = subplot(gridspec[r_i:r_i + 1, 0])
        title_ax.axis('off')

        r_i += 1
        target_ax = subplot(gridspec[r_i:r_i + 1, 0])

        r_i += 1
        features_ax = subplot(gridspec[r_i:r_i + features.shape[0], 0])

        r_i += features.shape[0]

        # Plot title
        title_ax.text(
            title_ax.axis()[1] / 2,
            -title_ax.axis()[2] / 2,
            '{} (n={})'.format(name, target.size),
            horizontalalignment='center',
            **FONT_LARGER)

        _plot_match(
            target,
            features,
            annotations,
            target_type,
            features_type,
            None,
            False,
            None,
            target_ax=target_ax,
            features_ax=features_ax)

        # Plot colorbar
        if r_i == n - 1:
            colorbar_ax = subplot(gridspec[r_i:r_i + 1, 0])
            colorbar_ax.axis('off')
            cax, kw = make_axes(
                colorbar_ax,
                location='bottom',
                pad=0.026,
                fraction=0.26,
                shrink=2.6,
                aspect=26,
                cmap=target_cmap,
                norm=Normalize(-3, 3),
                ticks=range(-3, 4, 1))
            ColorbarBase(cax, **kw)
    # Save
    save_plot(file_path)


def _prepare_data_for_plotting(a, data_type, max_std=3):
    """
    Prepare data (target | features) for plotting.
    Arguments:
         a (array): (n) | (n, m)
         data_type (str): 'continuous' | 'categorical' | 'binary'
         max_std (number):
    Returns:
         DataFrame:
         float:
         float:
         cmap:
    """

    if data_type == 'continuous':

        if a.ndim == 1:
            a_ = array_1d_normalize(a.values, method='-0-')
            a = Series(a_, name=a.name, index=a.index)

        elif a.ndim == 2:
            a_ = array_2d_normalize(a.values, method='-0-', axis=1)
            a = DataFrame(a_, index=a.index, columns=a.columns)

        return a, -max_std, max_std, CMAP_CONTINUOUS_ASSOCIATION

    elif data_type == 'categorical':
        return a.copy(), 0, unique(a).size, CMAP_CATEGORICAL_TAB20

    elif data_type == 'binary':
        return a.copy(), 0, 1, CMAP_BINARY_BW
