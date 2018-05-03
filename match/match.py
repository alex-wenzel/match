from math import ceil

from numpy import apply_along_axis, array_split, concatenate, full, nan
from numpy.random import choice, get_state, seed, set_state, shuffle
from pandas import DataFrame

from .nd_array.nd_array.compute_empirical_p_values_and_fdrs import \
    compute_empirical_p_values_and_fdrs
from .nd_array.nd_array.compute_nd_array_margin_of_error import \
    compute_nd_array_margin_of_error
from .nd_array.nd_array.drop_bad_value_and_apply_function_on_2_1d_arrays import \
    drop_bad_value_and_apply_function_on_2_1d_arrays
from .support.support.multiprocess import multiprocess
from .support.support.series import get_extreme_series_indices


def match(target,
          features,
          min_n_sample,
          match_function,
          n_job=1,
          extreme_feature_threshold=24,
          n_sampling=0,
          n_permutation=0,
          random_seed=20121020):

    results = DataFrame(columns=('Score', '0.95 MoE', 'P-Value', 'FDR'))

    n_job = min(features.shape[0], n_job)

    print('Computing score using {} with {} process{} ...'.format(
        match_function.__name__, n_job, ('', 'es')[1 < n_job]))

    results['Score'] = concatenate(
        multiprocess(match_target_and_features,
                     ((target, features_, min_n_sample, match_function)
                      for features_ in array_split(features, n_job)), n_job))

    if results['Score'].isna().all():

        raise ValueError(
            'Could not compute any score; perhaps because there were less than {} (min_n_sample) non-na values for all target-feature pairs to compute the score.'.
            format(min_n_sample))

    indices = get_extreme_series_indices(results['Score'],
                                         extreme_feature_threshold)

    if 3 <= n_sampling and 3 <= ceil(0.632 * target.size):

        results.loc[
            indices,
            '0.95 MoE'] = match_randomly_sampled_target_and_features_to_compute_margin_of_errors(
                target, features[indices], 3, match_function, n_sampling,
                random_seed)

    if 1 <= n_permutation:

        permutation_scores = concatenate(
            multiprocess(permute_target_and_match_target_and_features,
                         ((target, features_, min_n_sample, match_function,
                           n_permutation, random_seed)
                          for features_ in array_split(features, n_job)),
                         n_job))

        p_values, fdrs = compute_empirical_p_values_and_fdrs(
            results['Score'], permutation_scores.flatten())

        results['P-Value'] = p_values

        results['FDR'] = fdrs

    return results


def match_randomly_sampled_target_and_features_to_compute_margin_of_errors(
        target, features, min_n_sample, match_function, n_sampling,
        random_seed):

    if n_sampling < 3:

        raise ValueError('Cannot compute MoEs because n_sampling < 3.')

    if ceil(0.632 * target.size) < 3:

        raise ValueError('Cannot compute MoEs because 0.632 * n_sample < 3.')

    print('Computing MoE with {} samplings ...'.format(n_sampling))

    feature_x_sampling = full((features.shape[0], n_sampling), nan)

    seed(random_seed)

    for i in range(n_sampling):

        random_indices = choice(target.size, ceil(0.632 * target.size))

        sampled_target = target[random_indices]

        sampled_features = features[:, random_indices]

        random_state = get_state()

        feature_x_sampling[:, i] = match_target_and_features(
            sampled_target, sampled_features, min_n_sample, match_function)

        set_state(random_state)

    return apply_along_axis(compute_nd_array_margin_of_error, 1,
                            feature_x_sampling)


def permute_target_and_match_target_and_features(target, features,
                                                 min_n_sample, match_function,
                                                 n_permutation, random_seed):

    if n_permutation < 1:

        raise ValueError(
            'Not computing P-Value and FDR because n_permutation < 1.')

    print('Computing p-value and FDR with {} permutations ...'.format(
        n_permutation))

    feature_x_permutation = full((features.shape[0], n_permutation), nan)

    permuted_target = target.copy()

    seed(random_seed)

    for i in range(n_permutation):

        shuffle(permuted_target)

        random_state = get_state()

        feature_x_permutation[:, i] = match_target_and_features(
            permuted_target, features, min_n_sample, match_function)

        set_state(random_state)

    return feature_x_permutation


def match_target_and_features(target, features, min_n_sample, match_function):

    return apply_along_axis(
        drop_bad_value_and_apply_function_on_2_1d_arrays,
        1,
        features,
        target,
        min_n_sample,
        match_function,
        raise_=False)
