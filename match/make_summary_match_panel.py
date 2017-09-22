from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from pandas import DataFrame

from .array_nd.array_nd.array_2d import cluster_within_group
from .match import match
from .plot.plot.save_plot import save_plot
from .plot.plot.style import FIGURE_SIZE, FONT_LARGER, FONT_LARGEST
from .plot_match import plot_match
from .prepare_data_for_plotting import prepare_data_for_plotting

RANDOM_SEED = 20121020


def make_summary_match_panel(
        target,
        multiple_features,
        plot_only_columns_shared_by_target_and_all_features=False,
        target_ascending=False,
        n_samplings=30,
        n_permutations=30,
        random_seed=RANDOM_SEED,
        target_type='continuous',
        features_type='continuous',
        title='Summary Match Panel',
        plot_sample_names=False,
        file_path=None,
        dpi=100):
    """
    Make summary match panel.
    Arguments:
        target (Series): (n_samples)
        multiple_features (iterable): [
            Feature name (str):,
            Features (DataFrame): (n_features, n_samples),
            Scores (None | DataFrame): None (to compute match scores) |
                DataFrame (returned from make_match_panel)
            score_ascending (bool): True (scores increase from top to bottom) |
                False
            Index (iterable): Features to plot,
            Index alias (iterable): Name shown for the features to plot,
            Feature type (str): 'continuous' | 'categorical' | 'binary',
        ]
        plot_only_columns_shared_by_target_and_all_features (bool):
        target_ascending (bool): True if target increase from left to right,
            and False right to left
        title (str): Plot title
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        plot_sample_names (bool): Whether to plot column names
        file_path (str):
        dpi (int):
    Returns:
        None
    """

    # Set up figure
    fig = figure(figsize=FIGURE_SIZE)

    # Compute the number of row-grids for setting up a figure
    n = 0
    for f in multiple_features:
        n += len(f[5]) + 3

    # Set up axis grids
    gridspec = GridSpec(n, 1)

    # Annotate target with features
    r_i = 0
    fig.suptitle(title, horizontalalignment='center', **FONT_LARGEST)

    indexs = target.index
    if plot_only_columns_shared_by_target_and_all_features:
        for f in multiple_features:
            indexs &= f[1].columns

    for fi, (name, features, scores, scores_ascending, index, alias,
             features_type) in enumerate(multiple_features):

        print('Sorting target and features.columns ...')
        target = target.sort_values(ascending=target_ascending
                                    or target.dtype == 'O')
        features = features[target.index]

        if target.dtype == 'O':
            print('Making target numerical ...')
            target_o_to_int = {}
            target_int_to_o = {}
            for i, o in enumerate(target.unique()):
                target_o_to_int[o] = i
                target_int_to_o[i] = o
            target = target.map(target_o_to_int)

        if target_type in ('binary', 'categorical'):
            print('Clustering within categories ...')
            columns = cluster_within_group(target.values, features.values)
            features = features.iloc[:, columns]

        if scores:
            scores = scores.loc[index]
        else:
            print('Matching ...')
            scores = match(
                target.values,
                features.values,
                n_features=0,
                n_samplings=n_samplings,
                n_permutations=n_permutations,
                random_seed=random_seed)

        print('Sorting score ...')
        scores = scores.sort_values('Score', ascending=scores_ascending)
        features = features.loc[scores.index]

        # Use alias
        i_to_a = {i: a for i, a in zip(index, alias)}
        features.index = features.index.map(i_to_a)

        print('Making annotations ...')
        annotations = DataFrame(index=scores.index)

        # Make IC(confidence interval)
        annotations['IC(\u0394)'] = scores[['Score', '0.95 CI']].apply(
            lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)

        # Make p-value
        annotations['p-value'] = scores['p-value'].apply('{:.2e}'.format)

        # Make FDR
        annotations['FDR'] = scores['FDR'].apply('{:.2e}'.format)

        # Plot title
        r_i += 1
        title_ax = subplot(gridspec[r_i:r_i + 1, 0])
        title_ax.axis('off')

        title_ax.text(
            title_ax.axis()[1] / 2,
            0,
            '{} (n={})'.format(name, target.size),
            horizontalalignment='center',
            **FONT_LARGER)

        r_i += 1
        target_ax = subplot(gridspec[r_i:r_i + 1, 0])

        r_i += 1
        features_ax = subplot(gridspec[r_i:r_i + features.shape[0], 0])

        r_i += features.shape[0]

        # Plot match
        plot_match(
            target,
            target_int_to_o,
            features,
            annotations,
            None,
            target_type,
            features_type,
            None,
            plot_sample_names and fi == len(multiple_features) - 1,
            None,
            dpi,
            target_ax=target_ax,
            features_ax=features_ax)

    if file_path:
        save_plot(file_path)
