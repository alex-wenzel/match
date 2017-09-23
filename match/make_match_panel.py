from pandas import DataFrame

from .array_nd.array_nd.array_2d import cluster_within_group
from .match import match
from .plot_match import plot_match
from .support.support.path import establish_path
from .support.support.s import get_top_and_bottom_indexs

RANDOM_SEED = 20121020


def make_match_panel(target,
                     features,
                     target_ascending=False,
                     n_jobs=1,
                     scores_ascending=False,
                     n_features=0.99,
                     max_n_features=100,
                     n_samplings=30,
                     n_permutations=30,
                     random_seed=RANDOM_SEED,
                     figure_size=None,
                     title=None,
                     target_type='continuous',
                     features_type='continuous',
                     max_std=3,
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
        scores_ascending (bool): True (scores increase from top to bottom) |
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
        max_std (number):
        plot_sample_names (bool): Whether to plot column names
        file_path_prefix (str): file_path_prefix.match.txt and
            file_path_prefix.match.png will be saved
        dpi (int):
    Returns:
        DataFrame; (n_features, 4 ('Score', '<confidence_interval> CI',
            'p-value', 'FDR'))
    """

    if not target.index.symmetric_difference(features.columns).empty:
        raise ValueError(
            'target.index and features.columns have different object.')

    # Sort target and features.columns (based on target.index)
    target = target.sort_values(ascending=target_ascending
                                or target.dtype == 'O')
    features = features[target.index]

    target_o_to_int = {}
    target_int_to_o = {}
    if target.dtype == 'O':
        # Make target numerical
        for i, o in enumerate(target.unique()):
            target_o_to_int[o] = i
            target_int_to_o[i] = o
        target = target.map(target_o_to_int)

    if target_type in ('binary', 'categorical'):
        # Cluster within categories
        columns = cluster_within_group(target.values, features.values)
        features = features.iloc[:, columns]

    # Match
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

    # Sort scores
    scores.sort_values('Score', ascending=scores_ascending, inplace=True)

    if file_path_prefix:
        file_path_txt = file_path_prefix + '.match.txt'
        file_path_plot = file_path_prefix + '.match.png'
        # Save scores
        establish_path(file_path_txt)
        scores.to_csv(file_path_txt, sep='\t')
    else:
        file_path_plot = None

    # Select indexs to plot
    indexs = get_top_and_bottom_indexs(
        scores['Score'], n_features, max_n=max_n_features)

    scores_to_plot = scores.loc[indexs]
    features_to_plot = features.loc[scores_to_plot.index]

    # Make annotations
    annotations = DataFrame(index=scores_to_plot.index)
    # Make IC(confidence interval)s
    annotations['IC(\u0394)'] = scores_to_plot[['Score', '0.95 CI']].apply(
        lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)
    # Make p-values
    annotations['p-value'] = scores_to_plot['p-value'].apply('{:.2e}'.format)
    # Make FDRs
    annotations['FDR'] = scores_to_plot['FDR'].apply('{:.2e}'.format)

    # Plot match panel
    plot_match(target, target_int_to_o, features_to_plot, max_std, annotations,
               figure_size, None, None, target_type, features_type, title,
               plot_sample_names, file_path_plot, dpi)

    return scores
