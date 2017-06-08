from matplotlib.colorbar import ColorbarBase, make_axes
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from numpy import array, unique
from pandas import DataFrame, Series, read_table
from seaborn import heatmap

from .dataplay.dataplay.a import normalize as normalize_a
from .dataplay.dataplay.a2d import normalize as normalize_a2d
from .file.file.file import establish_path
from .helper.helper.df import (drop_slices_containing_only,
                               get_top_and_bottom_indices)
from .helper.helper.iterable import get_uniques_in_order
from .match import match
from .plot.plot.plot import save_plot
from .plot.plot.style import (CMAP_BINARY, CMAP_CATEGORICAL,
                              CMAP_CONTINUOUS_ASSOCIATION, FIGURE_SIZE,
                              FONT_LARGER, FONT_LARGEST, FONT_STANDARD,
                              decorate)

RANDOM_SEED = 20121020

SPACING = 0.05


def make_match_panel(target,
                     features,
                     dropna='all',
                     target_ascending=False,
                     min_n_unique_objects=2,
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
    Compute: scores[i] = function(target, features[i]); confidence intervals
    (CI) for n_features features; p-values; FDRs; and plot n_features features.
    :param target: Series; (n_samples); DataFrame must have columns matching
    features' columns
    :param features: DataFrame; (n_features, n_samples);
    :param dropna: str; 'all' | 'any'
    :param target_ascending: bool; True if target increase from left to right,
        and False right to left
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

    # Make sure target is a Series and features a DataFrame.
    # Keep samples found in both target and features.
    # Drop features with less than 2 unique values.
    target, features = _preprocess_target_and_features(
        target,
        features,
        dropna=dropna,
        target_ascending=target_ascending,
        min_n_unique_objects=min_n_unique_objects)

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

    # Save
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


def _preprocess_target_and_features(target, features, dropna, target_ascending,
                                    min_n_unique_objects):
    """
    Make sure target is a Series.
    Drop features with less than min_n_unique_objects unique values.
    Keep samples found in both target and features.
    :param target: iterable | Series
    :param features: DataFrame
    :param dropna: 'any' | 'all'
    :param target_ascending: bool
    :param min_n_unique_objects: int
    :return: Series & DataFrame
    """

    if not isinstance(target, Series):
        target = Series(target, index=features.columns)

    # Drop features having less than 2 unique values
    features = drop_slices_containing_only(
        features, min_n_unique_objects=min_n_unique_objects, axis=1)

    if features.empty:
        raise ValueError('No feature has at least {} unique objects.'.format(
            min_n_unique_objects))

    # Keep only columns shared by target and features
    shared = target.index & features.columns

    if len(shared):
        print(
            'Target ({} cols) and features ({} cols) have {} shared columns.'.
            format(target.size, features.shape[1], len(shared)))
        target = target.ix[shared].sort_values(ascending=target_ascending)
        features = features.ix[:, target.index]

    else:
        raise ValueError(
            'Target ({} cols) and features ({} cols) have 0 shared columns.'.
            format(target.size, features.shape[1]))

    return target, features


def _plot_match(target, features, annotations, target_type, features_type,
                title, plot_sample_names, file_path):
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

    # Prepare target & features for plotting
    target, target_min, target_max, target_cmap = _prepare_data_for_plotting(
        target, target_type)
    features, features_min, features_max, features_cmap = _prepare_data_for_plotting(
        features, features_type)

    # Set up figure
    figure(figsize=(min(pow(features.shape[1], 0.7), 7), pow(features.shape[0],
                                                             0.9)))

    # Set up grids & axes
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
            target_ax.axis()[1] * 0.5,
            target_ax.axis()[3] * 1.9,
            title,
            horizontalalignment='center',
            **FONT_LARGEST)

    # Plot annotation header
    target_ax.text(
        target_ax.axis()[1] + target_ax.axis()[1] * SPACING,
        target_ax.axis()[3] * 0.5,
        ' ' * 6 + 'IC(\u0394)' + ' ' * 10 + 'p-value' + ' ' * 12 + 'FDR',
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
            features_ax.axis()[3] - i - 0.5,
            '\t'.join(a.tolist()).expandtabs(),
            verticalalignment='center',
            **FONT_STANDARD)

    # Save
    if file_path:
        save_plot(file_path)


def make_summary_match_panel(target,
                             data_bundle,
                             annotation_files,
                             order=(),
                             target_ascending=False,
                             target_type='continuous',
                             title=None,
                             file_path=None):
    """
    Make summary match panel.    Make summary match panel
    :param target: Series; (n_elements);
    :param data_bundle: dict;
    :param annotation_files: dict;
    :param order: iterable;
    :param target_ascending: bool;
    :param target_type: str;
    :param title; str;
    :param file_path: str;
    :return: None
    """

    # Prepare target for plotting
    target, target_min, target_max, target_cmap = _prepare_data_for_plotting(
        target, target_type)

    #
    # Set up figure
    #
    # Compute the number of row-grids for setting up a figure
    n = 0
    for features_name, features_dict in data_bundle.items():
        n += features_dict['dataframe'].shape[0] + 3
    # Add a row for color bar
    n += 1
    # Set up figure
    fig = figure(figsize=FIGURE_SIZE)
    # Set up axis grids
    gridspec = GridSpec(n, 1)

    #
    # Annotate target with features
    #
    r_i = 0
    if not title:
        title = 'Association Summary Panel for {}'.format(title(target.name))
    fig.suptitle(title, horizontalalignment='center', **FONT_LARGEST)
    plot_annotation_header = True

    if not any(order):  # Sort alphabetically if order is not given
        order = sorted(data_bundle.keys())
    for features_name, features_dict in [(k, data_bundle[k]) for k in order]:

        # Read features
        features = features_dict['dataframe']

        # Prepare features for plotting
        features, features_min, features_max, features_cmap = _prepare_data_for_plotting(
            features, features_dict['data_type'])

        # Keep only columns shared by target and features
        shared = target.index & features.columns
        if any(shared):
            a_target = target.ix[shared].sort_values(
                ascending=target_ascending)
            features = features.ix[:, a_target.index]
            print(
                'Target {} ({} cols) and features ({} cols) have {} shared columns.'.
                format(target.name, target.size, features.shape[1],
                       len(shared)))
        else:
            raise ValueError(
                'Target {} ({} cols) and features ({} cols) have 0 shared column.'.
                format(target.name, target.size, features.shape[1]))

        # Read corresponding annotations file
        annotations = read_table(annotation_files[features_name], index_col=0)
        # Keep only features in the features dataframe and sort by score
        annotations = annotations.ix[features_dict['original_index'], :]
        annotations.index = features.index
        annotations.sort_values(
            'score', ascending=features_dict['emphasis'] == 'low')

        # Apply the sorted index to featuers
        features = features.ix[annotations.index, :]

        # TODO: update logic and consider removing this
        # if any(features_dict['alias']):  # Use alias as index
        #     features.index = features_dict['alias']
        #     annotations.index = features.index

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
            title_ax.axis()[1] * 0.5,
            title_ax.axis()[3] * 0.3,
            '{} (n={})'.format(title(features_name), len(shared)),
            horizontalalignment='center',
            **FONT_LARGER)

        # Plot target
        heatmap(
            DataFrame(a_target).T,
            ax=target_ax,
            vmin=target_min,
            vmax=target_max,
            cmap=target_cmap,
            xticklabels=False,
            yticklabels=True,
            cbar=False)
        for t in target_ax.get_yticklabels():
            t.set(rotation=0, **FONT_STANDARD)

        if plot_annotation_header:  # Plot header only for the 1st target axis
            target_ax.text(
                target_ax.axis()[1] + target_ax.axis()[1] * SPACING,
                target_ax.axis()[3] * 0.5,
                ' ' * 1 + 'IC(\u0394)' + ' ' * 6 + 'p-val' + ' ' * 15 + 'FDR',
                verticalalignment='center',
                **FONT_STANDARD)
            plot_annotation_header = False

        # Plot features
        heatmap(
            features,
            ax=features_ax,
            vmin=features_min,
            vmax=features_max,
            cmap=features_cmap,
            xticklabels=False,
            cbar=False)
        for t in features_ax.get_yticklabels():
            t.set(rotation=0, **FONT_STANDARD)

        # Plot annotations
        for i, (a_i, a) in enumerate(annotations.iterrows()):
            features_ax.text(
                features_ax.axis()[1] + features_ax.axis()[1] * SPACING,
                features_ax.axis()[3] - i *
                (features_ax.axis()[3] / features.shape[0]) - 0.5,
                '{0:.3f}\t{1:.2e}\t{2:.2e}'.format(*a.ix[
                    ['score', 'p-value', 'fdr']]).expandtabs(),
                verticalalignment='center',
                **FONT_STANDARD)

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
                ticks=[])
            ColorbarBase(cax, **kw)
            cax.text(
                cax.axis()[1] * 0.5,
                cax.axis()[3] * -2.6,
                'Standardized Profile for Target and Features',
                horizontalalignment='center',
                **FONT_STANDARD)
    # Save
    save_plot(file_path)


def _prepare_data_for_plotting(a, data_type, max_std=3):
    """
    Prepare data for plotting.
    :param a: array; (n) | (n, m)
    :param data_type: str; 'continuous' | 'categorical' | 'binary'
    :param max_std: number
    :return: DataFrame & float & float & cmap
    """

    if data_type == 'continuous':
        if a.ndim == 2:
            return DataFrame(
                normalize_a2d(
                    a, method='-0-', axis=1),
                index=a.index,
                columns=a.
                columns), -max_std, max_std, CMAP_CONTINUOUS_ASSOCIATION
        else:
            return Series(
                normalize_a(
                    a, method='-0-'),
                index=a.index), -max_std, max_std, CMAP_CONTINUOUS_ASSOCIATION

    elif data_type == 'categorical':
        return a.copy(), 0, unique(a).size, CMAP_CATEGORICAL

    elif data_type == 'binary':
        return a.copy(), 0, 1, CMAP_BINARY

    else:
        raise ValueError('Unknown data_type: {}.'.format(data_type))
