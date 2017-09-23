from math import ceil, sqrt

from numpy import (apply_along_axis, array, array_split, concatenate, empty,
                   isnan, where)
from numpy.random import choice, get_state, seed, set_state, shuffle
from pandas import DataFrame
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import multipletests

from .information.information.information import \
    compute_information_coefficient
from .support.support.multiprocess import multiprocess
from .support.support.s import get_top_and_bottom_indexs

RANDOM_SEED = 20121020


def match(target,
          features,
          function=compute_information_coefficient,
          n_jobs=1,
          n_features=0.99,
          max_n_features=100,
          n_samplings=30,
          confidence_interval=0.95,
          n_permutations=30,
          random_seed=RANDOM_SEED):
    """
    Compute: scores[i] = function(target, features[i]); confidence interval
        (CI) for n_features features; p-value; and FDR.
    Arguments:
        target (array): (n_samples)
        features (array): (n_features, n_samples)
        function (callable):
        n_jobs (int): Number of multiprocess jobs
        n_features (number): Number of features to compute CI and
            plot; number threshold if 1 <=, percentile threshold if < 1, and
            don't compute if None
        max_n_features (int):
        n_samplings (int): Number of bootstrap samplings to build distribution
            to get CI; must be 2 < to compute CI
        confidence_interval (float): CI
        n_permutations (int): Number of permutations for permutation test to
            compute p-values and FDR
        random_seed (int | array):
    Returns:
        DataFrame: (n_features, 4 ['Score', '<confidence_interval> CI',
            'p-value', 'FDR'])
    """

    results = DataFrame(columns=[
        'Score', '{} CI'.format(confidence_interval), 'p-value', 'FDR'
    ])

    # Compute scores[i] = function(target, features[i]) ...')
    def f(args):
        return match_target_and_features(*args)

    results['Score'] = concatenate(
        multiprocess(f, [(target, features, function)
                         for features in array_split(features, n_jobs)],
                     n_jobs))

    # Get top and bottom indexs
    indexs = get_top_and_bottom_indexs(
        results['Score'], n_features, max_n=max_n_features)

    # Compute CI
    if 3 <= n_samplings and 3 <= ceil(0.632 * target.size):

        results.loc[indexs, '{} CI'.format(
            confidence_interval)] = compute_confidence_interval(
                target,
                features[indexs],
                function,
                n_samplings=n_samplings,
                confidence_interval=confidence_interval,
                random_seed=random_seed)

    # Compute p-value and FDR
    if 1 <= n_permutations:

        permutation_scores = permute_and_match_target_and_features(
            target, features[indexs], function, n_permutations, random_seed)

        p_values, fdrs = compute_p_values_and_fdrs(
            results['Score'], permutation_scores.flatten())

        results['p-value'] = p_values
        results['FDR'] = fdrs

    return results


def compute_margin_of_errors(target,
                             features,
                             function,
                             n_samplings=30,
                             confidence_interval=0.95,
                             random_seed=RANDOM_SEED):
    """
    For n_samplings times, randomly choose 63.2% of the samples and match.
        Then compute margin of error.
    Arguments:
        target (array): (n_samples); 3 <= 0.632 * n_samples
        features (array): (n_features, n_samples)
        function (callable):
        n_samplings (int): 2 <
        cofidence (float):
        random_seed (int | array):
    Returns:
        array: (n)
    """

    if n_samplings < 3:
        raise ValueError('Cannot compute CI because n_samplings < 3.')

    if ceil(0.632 * target.size) < 3:
        raise ValueError('Cannot compute CI because 0.632 * n_samples < 3.')

    feature_x_sampling = empty((features.shape[0], n_samplings))

    seed(random_seed)
    for i in range(n_samplings):

        # Sample randomly
        random_indexs = choice(target.size, ceil(0.632 * target.size))
        sampled_target = target[random_indexs]
        sampled_features = features[:, random_indexs]

        random_state = get_state()

        # Score
        feature_x_sampling[:, i] = match_target_and_features(
            sampled_target, sampled_features, function)

        set_state(random_state)

    # Compute CI using bootstrapped score distributions
    return apply_along_axis(compute_margin_of_error, 1, feature_x_sampling)


def compute_margin_of_error(a, confidence=0.95):
    """
    Compute margin of error.
    Arguments:
        a (array):
        confidence (float):
    Returns:
        float
    """

    return norm.ppf(q=confidence) * a.std() / sqrt(a.size)


def permute_and_match_target_and_features(target,
                                          features,
                                          function,
                                          n_permutations=30,
                                          random_seed=RANDOM_SEED):
    """
    Permute target, remove indexs that are nan in either target or features[i]
        and compute: scores[i] = function(permuted_target, features[i]).
    Arguments:
        target (array): (n_samples)
        features (array): (n_features, n_samples)
        function (callable):
        n_permutations (int): 1 <=
        random_seed (int | array):
    Returns:
        array: (n_features, n_permutations)
    """

    if n_permutations < 1:
        raise ValueError(
            'Not computing p-value and FDR because n_permutations < 1.')

    feature_x_permutation = empty((features.shape[0], n_permutations))

    # Copy for inplace shuffling
    permuted_target = array(target)

    seed(random_seed)
    for i in range(n_permutations):

        # Permute
        shuffle(permuted_target)

        random_state = get_state()

        # Score
        feature_x_permutation[:, i] = match_target_and_features(
            permuted_target, features, function)

        set_state(random_state)

    return feature_x_permutation


def compute_p_values_and_fdrs(values, random_values):
    """
    Compute p-values and FDRs.
    Arguments:
        values (array): (n_features)
        random_values (array): (n_random_values)
    Returns:
        array: (n_features); p-values
        array: (n_features); FDRs
    """

    # Compute p-value
    p_values_l = array(
        [compute_p_value(v, random_values, 'left') for v in values])

    p_values_r = array(
        [compute_p_value(v, random_values, 'right') for v in values])

    # Take smaller p-value
    p_values = where(p_values_l < p_values_r, p_values_l, p_values_r)

    # Compute FDR
    fdrs_l = multipletests(p_values_l, method='fdr_bh')[1]
    fdrs_r = multipletests(p_values_r, method='fdr_bh')[1]

    # Take smaller FDR
    fdrs = where(fdrs_l < fdrs_r, fdrs_l, fdrs_r)

    return p_values, fdrs


def compute_p_value(value, random_values, direction):
    """
    Compute a p-value.
    Arguments:
        value (float):
        random_values (array):
        direction (str): 'left' | 'right'
    Returns:
        float: p-value
    """

    if direction == 'left':
        significant_random = random_values <= value

    elif direction == 'right':
        significant_random = value <= random_values

    else:
        raise ValueError('Unknown direction: {}.'.format(direction))

    p_value = significant_random.sum() / random_values.size

    if not p_value:
        p_value = 1 / random_values.size

    return p_value


def match_target_and_features(target, features, function):
    """
    Remove indexs that are nan in either target or features[i] and compute:
        scores[i] = function(target, features[i]).
    Arguments:
        target (array): (n_samples)
        features (array): (n_features, n_samples)
        function (callable):
    Returns:
        array: (n_features)
    """

    return apply_along_axis(remove_nans_and_match_a0_and_a1, 1, features,
                            target, function)


def remove_nans_and_match_a0_and_a1(a0, a1, function):
    """
    Remove indexs that are nan in either a0 or a1 and call function(a0, a1).
    Arguments:
        a0 (array): (n)
        a1 (array): (n)
        function (callable):
    Returns:
        float:
    """

    nans = isnan(a0) | isnan(a1)
    a0 = a0[~nans]
    a1 = a1[~nans]

    return function(a0.astype(float), a1.astype(float))
