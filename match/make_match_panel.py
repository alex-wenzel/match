from numpy import array
from pandas import DataFrame

from .match import match
from .plot_match import plot_match
from .preprocess_target_and_features import preprocess_target_and_features
from .support.support.df import get_top_and_bottom_indices
from .support.support.path import establish_path

RANDOM_SEED = 20121020


def make_match_panel(target,
                     features,
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
    Arguments:
        target (Series): (n_samples)
        features (DataFrame): (n_features, n_samples)
        target_ascending (bool): True if target increase from left to right,
            and False right to left
        max_n_unique_objects_for_drop_slices (int):
        result_in_ascending_order (bool): True if result increase from top to
            bottom, and False bottom to top
        n_jobs (int): Number of multiprocess jobs
        n_features (number): Number of features to compute CI and
            plot; number threshold if 1 <=, percentile threshold if < 1, and
            don't compute if None
        max_n_features (int):
        n_samplings (int): Number of bootstrap samplings to build distribution
            to get CI; must be 2 < to compute CI
        n_permutations (int): Number of permutations for permutation test to
            compute p-values and FDR
        random_seed (int | array):
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        title (str): Plot title
        plot_sample_names (bool): Whether to plot column names
        file_path_prefix (str): file_path_prefix.match.txt and
            file_path_prefix.match.pdf will be saved
    Returns:
        DataFrame; (n_features, 4 ('Score', '<confidence_interval> CI',
            'p-value', 'FDR'))
    """

    target, features = preprocess_target_and_features(
        target, features, target_ascending,
        max_n_unique_objects_for_drop_slices)

    print('Matching ...')
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
    plot_match(
        target,
        features_to_plot,
        annotations,
        target_type=target_type,
        features_type=features_type,
        title=title,
        plot_sample_names=plot_sample_names,
        file_path=file_path_pdf)

    return scores
