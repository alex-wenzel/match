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


def match(target,
          features,
          function=information_coefficient,
          n_jobs=1,
          n_features=0.95,
          n_samplings=30,
          confidence_interval=0.95,
          n_permutations=30,
          random_seed=RANDOM_SEED):
    """
    Compute: scores[i] = function(target, features[i]); confidence interval
        (CI) for n_features features; p-value; and FDR.
    :param target: array; (n_samples)
    :param features: array; (n_features, n_samples)
    :param function: callable
    :param n_jobs: int; number of multiprocess jobs
    :param n_features: number | None; number of features to compute CI; number
        threshold if 1 <=, percentile threshold if < 1, and don't compute if
        None
    :param n_samplings: int; number of bootstrap samplings to build
        distributions to get CI; must be 2 < to compute CI
    :param confidence_interval: float; CI
    :param n_permutations: int; number of permutations for computing p-value
        and FDR
    :param random_seed: int | array;
    :return: DataFrame; (n_features, 4 ('Score', '<confidence_interval> CI',
        'p-value', 'FDR'))
    """

    results = DataFrame(columns=[
        'Score',
        '{} CI'.format(confidence_interval),
        'p-value',
        'FDR',
    ])

    # Split features for parallel computing
    print('Using {} process{} ...'.format(n_jobs, ['es', ''][n_jobs == 1]))
    split_features = array_split(features, n_jobs)

    print('Computing scores[i] = function(target, features[i]) ...')

    results['Score'] = concatenate(
        multiprocess(multiprocess_score, [(target, fs, function)
                                          for fs in split_features], n_jobs))

    print('Computing {} CI ...'.format(confidence_interval))
    if n_samplings < 2:
        print('\tSkipping because n_samplings < 2.')
    elif ceil(0.632 * target.size) < 3:
        print('\tSkipping because 0.632 * n_samples < 3.')
    else:
        print('\tWith {} bootstrapped distributions ...'.format(n_samplings))

    indices = get_top_and_bottom_indices(results, 'Score', n_features)

    results.ix[indices, '{} CI'.format(
        confidence_interval)] = compute_confidence_interval(
            target,
            features[indices],
            function,
            n_samplings=n_samplings,
            confidence_interval=confidence_interval,
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
                      for f in split_features], n_jobs))

    p_values, fdrs = compute_p_values_and_fdrs(results['Score'],
                                               permutation_scores.flatten())

    results['p-value'] = p_values
    results['FDR'] = fdrs

    return results


def compute_p_values_and_fdrs(values, random_values):
    """
    Compute p-values and FDRs.
    :param values: array; (n_features)
    :param random_values: array; (n_random_values)
    :return array & array; (n_features) & (n_features); p-values & FDRs
    """

    # Compute p-values

    p_values_g = array([compute_p_value(v, random_values) for v in values])
    p_values_l = array(
        [compute_p_value(
            v, random_values, greater=False) for v in values])

    p_values = where(p_values_g < p_values_l, p_values_g, p_values_l)

    # Compute FDRs
    fdrs_g = multipletests(p_values_g, method='fdr_bh')[1]
    fdrs_l = multipletests(p_values_l, method='fdr_bh')[1]

    fdrs = where(fdrs_g < fdrs_l, fdrs_g, fdrs_l)

    return p_values, fdrs


def compute_p_value(value, random_values, greater=True):
    """
    Compute a p-value.
    :param value: float;
    :param random_values: array;
    :param greater: bool;
    :return: float; p-value
    """

    if greater:
        p_value = (value <= random_values).sum() / random_values.size
        if not p_value:
            p_value = 1 / random_values.size

    else:
        p_value = (random_values <= value).sum() / random_values.size
        if not p_value:
            p_value = 1 / random_values.size

    return p_value


def compute_confidence_interval(target,
                                features,
                                function=information_coefficient,
                                n_samplings=30,
                                confidence_interval=0.95,
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
        random_is = choice(target.size, ceil(0.632 * target.size))
        sampled_target = target[random_is]
        sampled_features = features[:, random_is]

        random_state = get_state()

        # Score
        feature_x_sampling[:, i] = apply_along_axis(
            lambda feature: function(sampled_target, feature), 1,
            sampled_features)

        set_state(random_state)

    # Compute CI using bootstrapped score distributions
    # TODO: Simplify calculation
    return apply_along_axis(
        lambda f: norm.ppf(q=confidence_interval) * f.std() / sqrt(n_samplings),
        1,
        feature_x_sampling)


def multiprocess_permute_and_score(args):
    """
    Permute_and_score for multiprocess mapping.
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
    Compute: scores[i] = function(permuted_target, features[i])
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
    Score for multiprocess mapping.
    :param args: iterable; (3)
    :return: array; (n_features, n_permutations)
    """

    return score(*args)


def score(target, features, function=information_coefficient):
    """
    Compute: scores[i] = function(permuted_target, features[i])
    :param target: array; (n_samples)
    :param features: array; (n_features, n_samples)
    :param function: callable
    :return: array; (n_features, n_permutations)
    """

    return apply_along_axis(lambda feature: function(target, feature), 1,
                            features)
