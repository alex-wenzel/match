from pandas import DataFrame

from .array_nd.array_nd.array_2d import cluster_within_group
from .match import match
from .plot_match import plot_match
from .support.support.df import get_top_and_bottom_indices
from .support.support.path import establish_path

RANDOM_SEED = 20121020


def make_match_panel(target,
                     features,
                     target_ascending=False,
                     n_jobs=1,
                     score_ascending=False,
                     n_features=0.99,
                     max_n_features=100,
                     n_samplings=30,
                     n_permutations=30,
                     random_seed=RANDOM_SEED,
                     figure_size=None,
                     title=None,
                     target_type='continuous',
                     features_type='continuous',
                     plot_sample_names=False,
                     file_path_prefix=None,
                     dpi=100):
    """
    Make match panel.
    Arguments:
        target (Series): (n_samples)
        features (DataFrame): (n_features, n_samples)
        target_ascending (bool): True if target increase from left to right,
            and False right to left
        n_jobs (int): Number of multiprocess jobs
        score_ascending (bool): True (scores increase from top to bottom) |
            False
        n_features (number): Number of features to compute CI and
            plot; number threshold if 1 <=, percentile threshold if < 1, and
            don't compute if None
        max_n_features (int):
        n_samplings (int): Number of bootstrap samplings to build distribution
            to get CI; must be 2 < to compute CI
        n_permutations (int): Number of permutations for permutation test to
            compute p-values and FDR
        random_seed (int | array):
        figure_size (tuple):
        title (str): Plot title
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        plot_sample_names (bool): Whether to plot column names
        file_path_prefix (str): file_path_prefix.match.txt and
            file_path_prefix.match.png will be saved
        dpi (int):
    Returns:
        DataFrame; (n_features, 4 ('Score', '<confidence_interval> CI',
            'p-value', 'FDR'))
    """

    if not target.index.symmetrix_difference(features.columns).empty:
        raise ValueError(
            'target.index and features.columns have different object.')

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

    print('Matching ...')
    scores = match(
        target.values,
        features.values,
        n_jobs=n_jobs,
        n_features=n_features,
        max_n_features=max_n_features,
        n_samplings=n_samplings,
        n_permutations=n_permutations,
        random_seed=random_seed)
    scores.index = features.index

    print('Sorting score ...')
    scores.sort_values('Score', ascending=score_ascending, inplace=True)

    if file_path_prefix:
        file_path_txt = file_path_prefix + '.match.txt'
        file_path_plot = file_path_prefix + '.match.png'

        print('Saving match results to {} ...'.format(file_path_txt))
        establish_path(file_path_txt)
        scores.to_csv(file_path_txt, sep='\t')

    else:
        file_path_plot = None

    # Keep only scores and features to plot
    indices = get_top_and_bottom_indices(
        scores, 'Score', n_features, max_n=max_n_features)

    scores_to_plot = scores.loc[indices]
    features_to_plot = features.loc[indices]

    print('Making annotations ...')
    annotations = DataFrame(index=scores_to_plot.index)

    # Make IC(confidence interval)
    annotations['IC(\u0394)'] = scores_to_plot[['Score', '0.95 CI']].apply(
        lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)

    # Make p-value
    annotations['p-value'] = scores_to_plot['p-value'].apply('{:.2e}'.format)

    # Make FDR
    annotations['FDR'] = scores_to_plot['FDR'].apply('{:.2e}'.format)

    print('Plotting match panel ...')
    plot_match(target, target_int_to_o, features_to_plot, annotations,
               figure_size, target_type, features_type, title,
               plot_sample_names, file_path_plot, dpi)

    return scores
