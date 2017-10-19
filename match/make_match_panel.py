from pandas import DataFrame

from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .match import match
from .nd_array.nd_array.cluster_2d_array_slices_by_group import \
    cluster_2d_array_slices_by_group
from .plot_match_panel import plot_match_panel
from .support.support.df import drop_df_slices
from .support.support.path import establish_path
from .support.support.series import get_top_and_bottom_series_indices

RANDOM_SEED = 20121020


def make_match_panel(target,
                     features,
                     target_ascending=False,
                     function=compute_information_coefficient,
                     n_jobs=1,
                     scores_ascending=False,
                     n_features=0.99,
                     max_n_features=100,
                     n_samplings=30,
                     n_permutations=30,
                     random_seed=RANDOM_SEED,
                     figure_size=None,
                     title='Match Panel',
                     target_type='continuous',
                     features_type='continuous',
                     max_std=3,
                     target_annotation_kwargs={'fontsize': 12},
                     plot_sample_names=False,
                     file_path_prefix=None,
                     dpi=100):
    """
    Make match panel.
    Arguments:
        target (Series): (n_samples); must be 3 <= 0.632 * n_samples to compute
            MoE
        features (DataFrame): (n_features, n_samples)
        target_ascending (bool): True if target increase from left to right,
            and False right to left
        function (callable): the function to use to compute similarity between
            target and features
        n_jobs (int): number of multiprocess jobs
        scores_ascending (bool): True (scores increase from top to bottom) |
            False
        n_features (number): number of features to compute MoE, p-value, and
            FDR; number threshold if 1 <= n_features, percentile threshold
            if n_features < 1, and don't compute if None
        max_n_features (int):
        n_samplings (int): number of bootstrap samplings to build distribution
            to compute MoE; 3 <= n_samplings
        n_permutations (int): number of permutations for permutation test to
            compute p-values and FDR
        random_seed (int | array):
        figure_size (iterable):
        title (str): plot title
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        max_std (number):
        target_annotation_kwargs (dict):
        plot_sample_names (bool): whether to plot column names
        file_path_prefix (str): file_path_prefix.match.txt and
            file_path_prefix.match.png will be saved
        dpi (int):
    Returns:
        DataFrame: (n_features, 4 ['Score', '<confidence> MoE', 'p-value',
            'FDR'])
    """

    # Sort target and features.columns (based on target)
    target = target.loc[target.index & features.columns].sort_values(
        ascending=target_ascending or target.dtype == 'O')

    # Drop constant rows
    features = drop_df_slices(
        features[target.index], 1, max_n_unique_objects=1)

    target_o_to_int = {}
    target_int_to_o = {}
    if target.dtype == 'O':
        # Make target numerical
        for i, o in enumerate(target.unique()):
            target_o_to_int[o] = i
            target_int_to_o[i] = o
        target = target.map(target_o_to_int)

    if target_type in ('binary', 'categorical'):
        # Cluster by group
        columns = cluster_2d_array_slices_by_group(features.values,
                                                   target.values)
        features = features.iloc[:, columns]

    # Match
    scores = match(
        target.values,
        features.values,
        function=function,
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

    # Select indices to plot
    indices = get_top_and_bottom_series_indices(scores['Score'], n_features)
    if max_n_features < indices.size:
        indices = indices[:max_n_features // 2].append(
            indices[-max_n_features // 2:])

    scores_to_plot = scores.loc[indices]
    features_to_plot = features.loc[scores_to_plot.index]

    # Make annotations
    annotations = DataFrame(index=scores_to_plot.index)
    # Make IC(MoE)s
    annotations['IC(\u0394)'] = scores_to_plot[['Score', '0.95 MoE']].apply(
        lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)
    # Make p-value
    annotations['p-value'] = scores_to_plot['p-value'].apply('{:.2e}'.format)
    # Make FDRs
    annotations['FDR'] = scores_to_plot['FDR'].apply('{:.2e}'.format)

    # Plot match panel
    plot_match_panel(target, target_int_to_o, features_to_plot, max_std,
                     annotations, figure_size, None, None, target_type,
                     features_type, title, target_annotation_kwargs,
                     plot_sample_names, file_path_plot, dpi)

    return scores
