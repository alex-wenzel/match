from pandas import DataFrame

from .match import match
from .plot_match import plot_match
from .preprocess_target_and_features import preprocess_target_and_features
from .support.support.df import get_top_and_bottom_indices
from .support.support.path import establish_path

RANDOM_SEED = 20121020


def make_match_panel(target,
                     features,
                     indexs=(),
                     target_ascending=False,
                     max_n_unique_objects_for_drop_slices=1,
                     increasing=False,
                     n_jobs=1,
                     n_features=0.99,
                     max_n_features=100,
                     n_samplings=30,
                     n_permutations=30,
                     random_seed=RANDOM_SEED,
                     figure_size=None,
                     target_type='continuous',
                     target_colormap=None,
                     features_type='continuous',
                     title=None,
                     plot_sample_names=False,
                     file_path_prefix=None):
    """
    Make match panel.
    Arguments:
        target (iterable): (n_samples)
        features (DataFrame): (n_features, n_samples)
        target_ascending (bool): True if target increase from left to right,
            and False right to left
        max_n_unique_objects_for_drop_slices (int):
        increasing (bool): True (scores increase from top to bottom) | False
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
        figure_size (tuple):
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

    target_o_to_int = {}
    target_int_to_o = {}

    if target.dtype == 'O':

        for i, o in enumerate(target.unique()):
            target_o_to_int[o] = i
            target_int_to_o[i] = o

        # Make target numerical
        target = target.map(target_o_to_int)

    target, features = preprocess_target_and_features(
        target, features, indexs, target_ascending,
        max_n_unique_objects_for_drop_slices)

    print('Matching ...')
    scores = match(
        target.values,
        features.values,
        n_jobs=n_jobs,
        n_features=n_features,
        n_samplings=n_samplings,
        n_permutations=n_permutations,
        random_seed=random_seed)

    scores.index = features.index

    scores.sort_values('Score', ascending=increasing, inplace=True)

    if file_path_prefix:
        file_path_txt = file_path_prefix + '.match.txt'
        file_path_pdf = file_path_prefix + '.match.png'
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

    # Add IC(confidence interval)
    annotations['IC(\u0394)'] = scores_to_plot[['Score', '0.95 CI']].apply(
        lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)

    # Add p-value
    annotations['p-value'] = scores_to_plot['p-value'].apply('{:.2e}'.format)

    # Add FDR
    annotations['FDR'] = scores_to_plot['FDR'].apply('{:.2e}'.format)

    print('Plotting match panel ...')
    plot_match(
        target,
        target_int_to_o,
        features_to_plot,
        annotations,
        figure_size,
        target_type,
        features_type,
        title,
        plot_sample_names,
        file_path_pdf,
        target_colormap=target_colormap)

    return scores
