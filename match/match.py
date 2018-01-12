from math import ceil

from numpy import apply_along_axis, array, array_split, concatenate, empty
from numpy.random import choice, get_state, seed, set_state, shuffle
from pandas import DataFrame

from .nd_array.nd_array.compute_empirical_p_values_and_fdrs import \
    compute_empirical_p_values_and_fdrs
from .nd_array.nd_array.compute_margin_of_error import compute_margin_of_error
from .nd_array.nd_array.drop_nan_and_apply_function_on_2_1d_arrays import \
    drop_nan_and_apply_function_on_2_1d_arrays
from .support.support.multiprocess import multiprocess
from .support.support.series import get_top_and_bottom_series_indices

RANDOM_SEED = 20121020


def match(target,
          features,
          min_n_samples,
          function,
          n_job=1,
          n_top_features=0.99,
          max_n_features=100,
          n_samplings=30,
          confidence=0.95,
          n_permutations=30,
          random_seed=RANDOM_SEED):
    """
    Compute: scores[i] = function(target, features[i]); compute margin of error
        (MoE), p-value, and FDR for n_top_features features.
    Arguments:
        target (array): (n_samples); must be 3 <= 0.632 * n_samples to compute
            MoE
        features (array): (n_features, n_samples)
        min_n_samples (int): the minimum number of samples needed for computing
        function (callable):
        n_job (int): number of multiprocess jobs
        n_top_features (number): number of features to compute MoE, p-value,
            and FDR; number threshold if 1 <= n_top_features and percentile
            threshold if 0.5 <= n_top_features < 1
        max_n_features (int):
        n_samplings (int): number of bootstrap samplings to build distribution
            to compute MoE; 3 <= n_samplings
        confidence (float):
        n_permutations (int): number of permutations for permutation test to
            compute p-values and FDR
        random_seed (int | array):
    Returns:
        DataFrame: (n_features, 4 ['Score', '<confidence> MoE', 'p-value',
            'FDR'])
    """

    results = DataFrame(
        columns=['Score', '{} MoE'.format(confidence), 'p-value', 'FDR'])

    # Match
    print('Computing match score with {} ({} process) ...'.format(
        function, n_job))

    results['Score'] = concatenate(
        multiprocess(match_target_and_features,
                     [(target, features_, min_n_samples, function)
                      for features_ in array_split(features, n_job)], n_job))

    # Get top and bottom indices
    indices = get_top_and_bottom_series_indices(results['Score'],
                                                n_top_features)
    if max_n_features and max_n_features < indices.size:
        indices = indices[:max_n_features // 2].append(
            indices[-max_n_features // 2:])

    # Compute MoE
    if 3 <= n_samplings and 3 <= ceil(0.632 * target.size):

        results.loc[indices, '{} MoE'.format(
            confidence
        )] = match_randomly_sampled_target_and_features_to_compute_margin_of_errors(
            target,
            features[indices],
            min_n_samples,
            function,
            n_samplings=n_samplings,
            confidence=confidence,
            random_seed=random_seed)

    # Compute p-value and FDR
    if 1 <= n_permutations:

        permutation_scores = concatenate(
            multiprocess(permute_target_and_match_target_and_features, [
                (target, features_, min_n_samples, function, n_permutations,
                 random_seed) for features_ in array_split(features, n_job)
            ], n_job))

        p_values, fdrs = compute_empirical_p_values_and_fdrs(
            results['Score'], permutation_scores.flatten())

        results['p-value'] = p_values
        results['FDR'] = fdrs

    return results


def match_randomly_sampled_target_and_features_to_compute_margin_of_errors(
        target,
        features,
        min_n_samples,
        function,
        n_samplings=30,
        confidence=0.95,
        random_seed=RANDOM_SEED):
    """
    Match randomly sampled target and features to compute margin of errors.
    Arguments
        target (array): (n_samples); must be 3 <= 0.632 * n_samples to compute
            MoE
        features (array): (n_features, n_samples)
        min_n_samples (int):
        function (callable):
        n_samplings (int): 3 <= n_samplings
        cofidence (float):
        random_seed (int | array):
    Returns:
        array: (n)
    """

    if n_samplings < 3:
        raise ValueError('Cannot compute MoEs because n_samplings < 3.')

    if ceil(0.632 * target.size) < 3:
        raise ValueError('Cannot compute MoEs because 0.632 * n_samples < 3.')

    print('Computing MoEs with {} samplings ...'.format(n_samplings))

    feature_x_sampling = empty((features.shape[0], n_samplings))

    seed(random_seed)
    for i in range(n_samplings):

        # Sample randomly
        random_indices = choice(target.size, ceil(0.632 * target.size))
        sampled_target = target[random_indices]
        sampled_features = features[:, random_indices]

        random_state = get_state()

        # Score
        feature_x_sampling[:, i] = match_target_and_features(
            sampled_target, sampled_features, min_n_samples, function)

        set_state(random_state)

    # Compute MoE using bootstrapped score distributions
    return apply_along_axis(compute_margin_of_error, 1, feature_x_sampling)


def permute_target_and_match_target_and_features(target,
                                                 features,
                                                 min_n_samples,
                                                 function,
                                                 n_permutations=30,
                                                 random_seed=RANDOM_SEED):
    """
    Permute target and match target and features.
    Arguments:
        target (array): (n_samples)
        features (array): (n_features, n_samples)
        min_n_samples (int):
        function (callable):
        n_permutations (int): 1 <= n_permutations
        random_seed (int | array):
    Returns:
        array: (n_features, n_permutations)
    """

    if n_permutations < 1:
        raise ValueError(
            'Not computing p-value and FDR because n_permutations < 1.')

    print('Computing p-values and FDRs with {} permutations ...'.format(
        n_permutations))

    feature_x_permutation = empty((features.shape[0], n_permutations))

    # Copy for inplace shuffling
    permuted_target = array(target)

    seed(random_seed)
    for i in range(n_permutations):
        if i % ceil(5000 / features.shape[0]) == 0:
            print('\t{}/{} ...'.format(i + 1, n_permutations))

        # Permute
        shuffle(permuted_target)

        random_state = get_state()

        # Match
        feature_x_permutation[:, i] = match_target_and_features(
            permuted_target, features, min_n_samples, function)

        set_state(random_state)
    print('\t{}/{} - done.'.format(i + 1, n_permutations))

    return feature_x_permutation


def match_target_and_features(target, features, min_n_samples, function):
    """
    Drop nan from target and features[i] and compute: scores[i] = function(
        target, features[i]).
    Arguments:
        target (array): (n_samples)
        features (array): (n_features, n_samples)
        min_n_samples (int):
        function (callable):
    Returns:
        array: (n_features)
    """

    return apply_along_axis(drop_nan_and_apply_function_on_2_1d_arrays, 1,
                            features, target, min_n_samples, function)
