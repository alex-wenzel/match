from math import ceil, sqrt

from numpy import (apply_along_axis, array, array_split, concatenate, empty,
                   where)
from numpy.random import choice, get_state, seed, set_state, shuffle
from pandas import DataFrame
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import multipletests

from .helper.helper.df import get_top_and_bottom_indices
from .helper.helper.multiprocess import multiprocess
from .information.information.information import information_coefficient

RANDOM_SEED = 20121020

# def make_match_panel(target,
#                      features,
#                      dropna='all',
#                      file_path_scores=None,
#                      target_ascending=False,
#                      result_in_ascending_order=False,
#                      n_jobs=1,
#                      n_features=0.95,
#                      max_n_features=100,
#                      n_samplings=30,
#                      n_permutations=30,
#                      random_seed=RANDOM_SEED,
#                      target_type='continuous',
#                      features_type='continuous',
#                      title=None,
#                      plot_column_names=False,
#                      file_path_prefix=None):
#     """
#     Compute: score_i = function(target, feature_i) for all features. Compute
#     confidence interval (CI) for n_features features. Compute p-value and FDR
#     (BH) for all features. And plot the result.
#     :param target: Series; (n_samples); must have index matching features'
#     columns
#     :param features: DataFrame; (n_features, n_samples);
#     :param dropna: str; 'any' | 'all'
#     :param file_path_scores: str;
#     :param target_ascending: bool;
#     :param result_in_ascending_order: bool; True if result increase from top to
#     bottom, and False otherwise
#     :param n_jobs: int; number of jobs for parallelizing
#     :param n_features: int or float; number threshold if >= 1, and percentile
#     threshold if < 1
#     :param max_n_features: int;
#     :param n_samplings: int; number of boGotstrap samplings to build distribution
#     to get CI; must be > 2 to compute CI
#     :param n_permutations: int; number of permutations for permutation test to
#     compute p-val and FDR
#     :param random_seed: int | array;
#     :param target_type: str; 'continuous' | 'categorical' | 'binary'
#     :param features_type: str; 'continuous' | 'categorical' | 'binary'
#     :param title: str; plot title
#     :param plot_column_names: bool; plot column names below the plot or not
#     :param file_path_prefix: str; file_path_prefix.match.txt and
#     file_path_prefix.match.pdf will be saved
#     :return: DataFrame; (n_features, 4 ('Score', '<confidence> MoE', 'p-value',
#     'FDR'))
#     """
#
#     if file_path_scores:  # Read pre-computed scores
#         print(
#             'Using precomputed scores (could have been calculated with a different number of samples) ...'
#         )
#
#         scores = read_table(file_path_scores, index_col=0)
#
#     else:  # Compute scores
#         scores = match()
#         scores.sort_values(
#             'Score', ascending=result_in_ascending_order, inplace=True)
#
#         # Save
#         if file_path_prefix:
#             file_path = file_path_prefix + '.match.txt'
#             establish_path(file_path)
#             scores.to_csv(file_path, sep='\t')
#


def match(target,
          features,
          function=information_coefficient,
          dropna='all',
          target_ascending=False,
          n_jobs=1,
          result_in_ascending_order=False,
          n_features=0.95,
          n_samplings=30,
          confidence=0.95,
          n_permutations=30,
          random_seed=RANDOM_SEED):
    """
    Compute: score[i] = function(target, features[i]); confidence interval (CI)
    for n_features features; p-value; and FDR.
    :param target: array; (n_samples)
    :param features: array; (n_features, n_samples)
    :param function: callable
    :param n_jobs: int; number of multiprocess jobs
    :param n_features: number | None; number of features to compute CI; number
    threshold if 1 <=, percentile threshold if < 1, and don't compute if None
    :param n_samplings: int; number of bootstrap samplings to build
    distributions to get CI; must be 2 < to compute CI
    :param confidence: float; CI confidence
    :param n_permutations: int; number of permutations for computing p-value
    and FDR
    :param random_seed: int | array;
    :return: DataFrame; (n_features, 4 ('Score', '<confidence> CI', 'p-value',
    'FDR'))
    """

    results = DataFrame(columns=[
        'Score',
        '{} CI'.format(confidence),
        'p-value (forward)',
        'p-value (reverse)',
        'p-value',
        'FDR (forward)',
        'FDR (reverse)',
        'FDR',
    ])

    # Split features for parallel computing
    print('Using {} process{} ...'.format(n_jobs, ['es', ''][n_jobs == 1]))
    split_features = array_split(features, n_jobs)

    print('Computing scores[i] = function(target, features[i]) ...')

    results['Score'] = concatenate(
        multiprocess(multiprocess_score, [(target, fs, function)
                                          for fs in split_features], n_jobs))

    print('Computing {} CI ...'.format(confidence))
    if n_samplings < 2:
        print('\tSkipping because n_samplings < 2.')
    elif ceil(0.632 * target.size) < 3:
        print('\tSkipping because 0.632 * n_samples < 3.')
    else:
        print('\tWith {} bootstrapped distributions ...'.format(
            n_samplings))

    indices = get_top_and_bottom_indices(results, 'Score', n_features)

    results.ix[indices, '{} CI'.format(
        confidence)] = compute_confidence_interval(
            target,
            features[indices],
            function,
            n_samplings=n_samplings,
            confidence=confidence,
            random_seed=random_seed)

    print('Computing p-value and FDR ...')
    if n_permutations < 1:
        print('\tSkipping because n_perm < 1.')
    else:
        print('\tBy scoring against {} permuted targets ...'.format(
            n_permutations))

        # Permute and score
        permutation_scores = concatenate(
            multiprocess(multiprocess_permute_and_score,
                         [(target, f, function, n_permutations, random_seed)
                          for f in split_features], n_jobs)).flatten()

        for i, (r_i, r) in enumerate(results.iterrows()):

            # This feature's score
            s = r.ix['Score']

            # Compute forward p-value
            p_value_forward = (
                s <= permutation_scores).sum() / permutation_scores.size
            if not p_value_forward:
                p_value_forward = 1 / permutation_scores.size
            results.ix[r_i, 'p-value (forward)'] = p_value_forward

            # Compute reverse p-value
            p_value_reverse = (
                permutation_scores <= s).sum() / permutation_scores.size
            if not p_value_reverse:
                p_value_reverse = 1 / permutation_scores.size
            results.ix[r_i, 'p-value (reverse)'] = p_value_reverse

        # Compute forward FDR
        results['FDR (forward)'] = multipletests(
            results['p-value (forward)'], method='fdr_bh')[1]

        # Compute reverse FDR
        results['FDR (reverse)'] = multipletests(
            results['p-value (reverse)'], method='fdr_bh')[1]

        # Creating the summary p-value and FDR
        f = results['p-value (forward)']
        r = results['p-value (reverse)']
        results['p-value'] = where(f < r, f, r)

        f = results['FDR (forward)']
        r = results['FDR (reverse)']
        results['FDR'] = where(f < r, f, r)

    return results


def compute_confidence_interval(target,
                                features,
                                function=information_coefficient,
                                n_samplings=30,
                                confidence=0.95,
                                random_seed=RANDOM_SEED):
    """
    For n_samplings times, randomly choose 63.2% of the samples, score, build
    score distribution, and compute CI.
    :param target: array; (n_samples)
    :param features: array; (n_features, n_samples)
    :param function: callable
    :param n_samplings int;
    :param cofidence: float;
    :param random_seed: int | array;
    :return: array; (n)
    """

    feature_x_sampling = empty((features.shape[0], n_samplings))

    seed(random_seed)
    for i in range(n_samplings):

        # Sample
        random_samples = choice(target.size, ceil(0.632 * target.size))
        sampled_target = target[random_samples]
        sampled_features = features[:, random_samples]

        random_state = get_state()

        # Score
        feature_x_sampling[:, i] = apply_along_axis(
            lambda f: function(sampled_target, f), 1, sampled_features)

        set_state(random_state)

    # Compute CI using bootstrapped score distributions
    # TODO: improve confidence interval calculation
    return apply_along_axis(
        lambda f: norm.ppf(q=confidence) * f.std() / sqrt(n_samplings),
        1,
        feature_x_sampling)


def multiprocess_permute_and_score(args):
    """
    Call permute_and_score; for multiprocess mapping.
    :param args: iterable; (5)
    :return: array; (n_features, n_permutations)
    """

    return permute_and_score(*args)


def permute_and_score(target,
                      features,
                      function=information_coefficient,
                      n_permutations=30,
                      random_seed=RANDOM_SEED):
    """
    Compute: score[i] = function(permuted_target, features[i])
    :param target: array; (n_samples)
    :param features: array; (n_features, n_samples)
    :param function: callable
    :param n_permutations: int;
    :param random_seed: int | array;
    :return: array; (n_features, n_permutations)
    """

    feature_x_permutation = empty((features.shape[0], n_permutations))

    # TODO: Speed up

    # Copy for inplace shuffling
    target = array(target)

    seed(random_seed)
    for i in range(n_permutations):

        # Permute
        shuffle(target)

        random_state = get_state()

        # Score
        feature_x_permutation[:, i] = score(
            target, features, function=function)

        set_state(random_state)

    return feature_x_permutation


def multiprocess_score(args):
    """
    Call score; for multiprocess mapping.
    :param args: iterable; (3)
    :return: array; (n_features, n_permutations)
    """

    return score(*args)


def score(target, features, function=information_coefficient):
    """
    Compute: score[i] = function(permuted_target, features[i])
    :param target: array; (n_samples)
    :param features: array; (n_features, n_samples)
    :param function: callable
    :return: array; (n_features, n_permutations)
    """

    return apply_along_axis(lambda f: function(target, f), 1, features)
