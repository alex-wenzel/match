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

    print('Matching using {} process ...'.format(n_jobs))
    split_features = array_split(features, n_jobs)

    print('Computing scores[i] = function(target, features[i]) ...')
    results['Score'] = concatenate(
        multiprocess(multiprocess_score, [(target, fs, function)
                                          for fs in split_features], n_jobs))

    print('Computing {} CI ...'.format(confidence_interval))
    if n_samplings < 2:
        print('\tskipped because n_samplings < 2.')
    elif ceil(0.632 * target.size) < 3:
        print('\tskipped because 0.632 * n_samples < 3.')
    else:
        print('\twith {} bootstrapped distributions ...'.format(n_samplings))

    indexs = get_top_and_bottom_indexs(
        results['Score'], n_features, max_n=max_n_features)

    results.loc[indexs, '{} CI'.format(
        confidence_interval)] = compute_confidence_interval(
            target,
            features[indexs],
            function,
            n_samplings=n_samplings,
            confidence_interval=confidence_interval,
            random_seed=random_seed)

    print('Computing p-value and FDR ...')
    if n_permutations < 1:
        print('\tskipped because n_perm < 1.')
    else:
        print('\tby scoring against {} permuted targets ...'.format(
            n_permutations))

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
    Arguments:
        values (array): (n_features)
        random_values (array): (n_random_values)
    Returns:
        array: (n_features); p-values
        array: (n_features); FDRs
    """

    # Compute p-value
    p_values_g = array([compute_p_value(v, random_values) for v in values])
    p_values_l = array(
        [compute_p_value(v, random_values, greater=False) for v in values])

    p_values = where(p_values_g < p_values_l, p_values_g, p_values_l)

    # Compute FDR
    fdrs_g = multipletests(p_values_g, method='fdr_bh')[1]
    fdrs_l = multipletests(p_values_l, method='fdr_bh')[1]

    fdrs = where(fdrs_g < fdrs_l, fdrs_g, fdrs_l)

    return p_values, fdrs


def compute_p_value(value, random_values, greater=True):
    """
    Compute a p-value.
    Arguments:
        value (float):
        random_values (array):
        greater (bool):
    Returns:
        float: p-value
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
                                function,
                                n_samplings=30,
                                confidence_interval=0.95,
                                random_seed=RANDOM_SEED):
    """
    For n_samplings times, randomly choose 63.2% of the samples, score, build
        score distribution, and compute CI.
    Arguments:
        target (array): (n_samples)
        features (array): (n_features, n_samples)
        function (callable):
        n_samplings (int):
        cofidence (float):
        random_seed (int | array):
    Returns:
        array: (n)
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
    Arguments:
        args (iterable): (5); permute_and_score's arguments
    Returns:
        array: (n_features, n_permutations)
    """

    return permute_and_score(*args)


def permute_and_score(target,
                      features,
                      function,
                      n_permutations=30,
                      random_seed=RANDOM_SEED):
    """
    Compute: scores[i] = function(permuted_target, features[i])
    Arguments:
        target (array): (n_samples)
        features (array): (n_features, n_samples)
        function (callable):
        n_permutations (int):
        random_seed (int | array):
    Returns:
        array: (n_features, n_permutations)
    """

    feature_x_permutation = empty((features.shape[0], n_permutations))

    # Copy for inplace shuffling
    target = array(target)

    seed(random_seed)
    for i in range(n_permutations):

        # Permute
        shuffle(target)

        random_state = get_state()

        # Score
        feature_x_permutation[:, i] = score(target, features, function)

        set_state(random_state)

    return feature_x_permutation


def multiprocess_score(args):
    """
    Score for multiprocess mapping.
    Arguments:
        args (iterable): (3); score's arguments
    Returns:
        array: (n_features)
    """

    return score(*args)


def score(target, features, function):
    """
    Compute: scores[i] = function(target, features[i])
    Arguments:
        target (array): (n_samples)
        features (array): (n_features, n_samples)
        function (callable):
    Returns:
        array: (n_features)
    """

    def f(x, y):
        """
        """
        # Drop indices with missing value in either x or y
        nans = isnan(x) | isnan(y)
        x = x[~nans]
        y = y[~nans]
        return function(x, y)

    return apply_along_axis(f, 1, features)
