from numpy import nan_to_num
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
                     scores=None,
                     min_n_sample=3,
                     function_=compute_information_coefficient,
                     n_job=1,
                     scores_ascending=False,
                     n_top_feature=25,
                     max_n_feature=100,
                     n_sampling=3,
                     n_permutation=3,
                     random_seed=RANDOM_SEED,
                     indices=None,
                     figure_size=None,
                     title='Match Panel',
                     target_type='continuous',
                     features_type='continuous',
                     max_std=3,
                     target_annotation_kwargs=None,
                     plot_column_names=False,
                     max_ytick_size=26,
                     file_path_prefix=None):
    """
    Make match panel.
    Arguments:
        target (Series): (n_sample, ); must be 3 <= 0.632 * n_sample to compute
            MoE
        features (DataFrame): (n_feature, n_sample, )
        target_ascending (bool): True if target increase from left to right,
            and False right to left
        min_n_sample (int):
        function_ (callable): function_ for computing match scores between the
            target and each feature
        scores (DataFrame): (n_feature, 4 ('Score', '<confidence> MoE',
            'p-value', 'FDR', ), )
        n_job (int): number of multiprocess jobs
        scores_ascending (bool): True (scores increase from top to bottom) |
            False
        n_top_feature (float): number of features to compute MoE, p-value,
            and FDR; number threshold if 1 <= n_top_feature and percentile
            threshold if 0.5 <= n_top_feature < 1
        max_n_feature (int):
        n_sampling (int): number of bootstrap samplings to build distribution
            to compute MoE; 3 <= n_sampling
        n_permutation (int): number of permutations for permutation test to
            compute p-values and FDR
        random_seed (float):
        indices (iterable):
        figure_size (iterable):
        title (str): plot title
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        max_std (float):
        target_annotation_kwargs (dict):
        plot_column_names (bool): whether to plot column names
        max_ytick_size (int):
        file_path_prefix (str): file_path_prefix.match.tsv and
            file_path_prefix.match.pdf will be saved
    Returns:
        DataFrame: (n_feature, 4 ('Score', '<confidence> MoE', 'p-value',
            'FDR', ), )
    """

    if target_annotation_kwargs is None:
        target_annotation_kwargs = {
            'fontsize': 12,
        }

    # Sort target and features.columns (based on target)
    target = target.loc[target.index & features.columns].sort_values(
        ascending=target_ascending or target.dtype == 'O')
    features = features[target.index]

    # Drop constant rows
    features = drop_df_slices(features, 1, max_n_unique_object=1)

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
        columns = cluster_2d_array_slices_by_group(
            nan_to_num(features.values), nan_to_num(target.values))
        features = features.iloc[:, columns]

    if scores is None:
        # Match
        scores = match(
            target.values,
            features.values,
            min_n_sample,
            function_,
            n_job=n_job,
            n_top_feature=n_top_feature,
            max_n_feature=max_n_feature,
            n_sampling=n_sampling,
            n_permutation=n_permutation,
            random_seed=random_seed)
        scores.index = features.index

        # Sort scores
        scores.sort_values('Score', ascending=scores_ascending, inplace=True)

        if file_path_prefix:
            # Save scores
            file_path_tsv = file_path_prefix + '.match.tsv'
            establish_path(file_path_tsv, 'file')
            scores.to_csv(file_path_tsv, sep='\t')

    # Select indices to plot
    if indices is None:
        indices = get_top_and_bottom_series_indices(scores['Score'],
                                                    n_top_feature)
        if max_n_feature and max_n_feature < indices.size:
            indices = indices[:max_n_feature // 2].append(
                indices[-max_n_feature // 2:])
    else:
        indices = sorted(
            indices,
            key=lambda j: scores.loc[j, 'Score'],
            reverse=not scores_ascending)
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
    if file_path_prefix:
        file_path_plot = file_path_prefix + '.match.pdf'
    else:
        file_path_plot = None
    plot_match_panel(target, target_int_to_o, features_to_plot, max_std,
                     annotations, figure_size, None, None, target_type,
                     features_type, title, target_annotation_kwargs,
                     plot_column_names, max_ytick_size, file_path_plot)

    return scores
