from numpy import nan_to_num
from pandas import DataFrame, Index

from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .match import match
from .nd_array.nd_array.cluster_2d_array_slices_by_group import \
    cluster_2d_array_slices_by_group
from .plot_match_panel import plot_match_panel
from .support.support.df import drop_df_slices
from .support.support.path import establish_path
from .support.support.series import get_top_and_bottom_series_indices


def make_match_panel(target,
                     features,
                     target_ascending=False,
                     scores=None,
                     min_n_sample=5,
                     function_=compute_information_coefficient,
                     random_seed=20121020,
                     n_job=1,
                     scores_ascending=False,
                     indices=None,
                     n_top_feature=10,
                     max_n_feature=100,
                     n_sampling=10,
                     n_permutation=10,
                     target_type='continuous',
                     features_type='continuous',
                     title='Match Panel',
                     max_ytick_size=50,
                     plot_column_names=False,
                     file_path_prefix=None):
    """
    Make match panel.
    Arguments:
        target (Series): (n_sample, ); must be 3 <= 0.632 * n_sample to compute
            MoE
        features (DataFrame): (n_feature, n_sample, )
        target_ascending (bool | None): True if target increase from left to
            right | False right to left | None for using the target's order
        scores (DataFrame): (n_feature, 4 ('Score', '<confidence> MoE',
            'P-Value', 'FDR', ), )
        min_n_sample (int):
        function_ (callable): function_ for computing match scores between the
            target and each feature
        random_seed (float):
        n_job (int): number of multiprocess jobs
        scores_ascending (bool): True (scores increase from top to bottom) |
            False
        indices (iterable):
        n_top_feature (float): number of features to compute MoE, P-Value,
            and FDR; number threshold if 1 <= n_top_feature and percentile
            threshold if 0.5 <= n_top_feature < 1
        max_n_feature (int):
        n_sampling (int): number of bootstrap samplings to build distribution
            to compute MoE; 3 <= n_sampling
        n_permutation (int): number of permutations for permutation test to
            compute P-Value and FDR
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        title (str): plot title
        max_ytick_size (int):
        plot_column_names (bool): whether to plot column names
        file_path_prefix (str): file_path_prefix.match.tsv and
            file_path_prefix.match.pdf will be saved
    Returns:
        DataFrame: (n_feature, 4 ('Score', '<confidence> MoE', 'P-Value',
            'FDR', ), )
    """

    target = target.loc[target.index & features.columns]

    if isinstance(target_ascending, bool):
        target.sort_values(ascending=target_ascending, inplace=True)

    features = features[target.index]

    features = drop_df_slices(features, 1, max_n_unique_object=1)

    if target_type in (
            'binary',
            'categorical', ):

        target_values = target.values.tolist()
        if all(((1 < target_values.count(i)) for i in target_values)):

            features = features.iloc[:,
                                     cluster_2d_array_slices_by_group(
                                         nan_to_num(features.values),
                                         nan_to_num(target.values))]

    if scores is None:
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

        if file_path_prefix:
            file_path_tsv = file_path_prefix + '.match.tsv'
            establish_path(file_path_tsv, 'file')
            scores.to_csv(file_path_tsv, sep='\t')

    if indices is None:
        indices = get_top_and_bottom_series_indices(scores['Score'],
                                                    n_top_feature).tolist()

    indices = Index(
        sorted(
            indices,
            key=lambda index: scores.loc[index, 'Score'],
            reverse=not scores_ascending))

    if max_n_feature and max_n_feature < indices.size:
        indices = indices[:max_n_feature // 2].append(
            indices[-max_n_feature // 2:])

    scores_to_plot = scores.loc[indices]
    features_to_plot = features.loc[scores_to_plot.index]

    annotations = DataFrame(index=scores_to_plot.index)
    annotations['IC(\u0394)'] = scores_to_plot[[
        'Score',
        '0.95 MoE',
    ]].apply(
        lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)
    annotations['P-Value'] = scores_to_plot['P-Value'].apply('{:.2e}'.format)
    annotations['FDR'] = scores_to_plot['FDR'].apply('{:.2e}'.format)

    if file_path_prefix:
        file_path_plot = file_path_prefix + '.match.pdf'
    else:
        file_path_plot = None

    plot_match_panel(target, features_to_plot, target_type, features_type,
                     None, None, title, None, max_ytick_size, annotations,
                     plot_column_names, file_path_plot)

    return scores
