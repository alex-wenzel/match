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


def match(target,
          features,
          min_n_sample,
          match_function,
          n_job=1,
          n_top_feature=0.99,
          max_n_feature=100,
          n_sampling=10,
          n_permutation=10,
          random_seed=20121020):
    """
    Compute: scores[i] = match_function_(target, features[i]) and margin of
    error (MoE), P-Value, and FDR for n_top_feature features.
    Arguments:
        target (ndarray): (n_sample, ); must be 3 <= 0.632 * n_sample to compute
            MoE
        features (ndarray): (n_feature, n_sample, )
        min_n_sample (int): the minimum number of samples needed for computing
        match_function (callable):
        n_job (int): number of multiprocess jobs
        n_top_feature (float): number of features to compute MoE, P-Value,
            and FDR; number threshold if 1 <= n_top_feature and percentile
            threshold if 0.5 <= n_top_feature < 1
        max_n_feature (int):
        n_sampling (int): number of bootstrap samplings to build distribution
            to compute MoE; 3 <= n_sampling
        n_permutation (int): number of permutations for permutation test to
            compute P-Value and FDR
        random_seed (float):
    Returns:
        DataFrame: (n_feature, 4 ('Score', '0.95 MoE', 'P-Value', 'FDR', ), )
    """

    results = DataFrame(columns=(
        'Score',
        '0.95 MoE',
        'P-Value',
        'FDR', ))

    if 1 < n_job:
        n_job = min(features.shape[0], n_job)

    print('Computing match score with {} ({} process) ...'.format(
        match_function, n_job))
    results['Score'] = concatenate(
        multiprocess(match_target_and_features, ((
            target,
            features_,
            min_n_sample,
            match_function, ) for features_ in array_split(features, n_job)),
                     n_job))

    if results['Score'].isna().all():
        raise ValueError(
            'Could not compute any score; perhaps because there were less than {} (min_n_sample) non-na values for all target-feature pairs to compute the score.'.
            format(min_n_sample))

    indices = get_top_and_bottom_series_indices(results['Score'],
                                                n_top_feature)
    if max_n_feature and max_n_feature < indices.size:
        indices = indices[:max_n_feature // 2].append(
            indices[-max_n_feature // 2:])

    if 3 <= n_sampling and 3 <= ceil(0.632 * target.size):
        results.loc[
            indices,
            '0.95 MoE'] = match_randomly_sampled_target_and_features_to_compute_margin_of_errors(
                target, features[indices], 3, match_function, n_sampling,
                random_seed)

    if 1 <= n_permutation:
        permutation_scores = concatenate(
            multiprocess(permute_target_and_match_target_and_features, ((
                target,
                features_,
                min_n_sample,
                match_function,
                n_permutation,
                random_seed, ) for features_ in array_split(features, n_job)),
                         n_job))

        p_values, fdrs = compute_empirical_p_values_and_fdrs(
            results['Score'], permutation_scores.flatten())
        results['P-Value'] = p_values
        results['FDR'] = fdrs

    return results


def match_randomly_sampled_target_and_features_to_compute_margin_of_errors(
        target, features, min_n_sample, match_function, n_sampling,
        random_seed):
    """
    Match randomly sampled target and features to compute margin of errors.
    Arguments
        target (ndarray): (n_sample, ); must be 3 <= 0.632 * n_sample to compute
            MoE
        features (ndarray): (n_feature, n_sample, )
        min_n_sample (int):
        match_function (callable):
        n_sampling (int): 3 <= n_sampling
        random_seed (float):
    Returns:
        ndarray: (n, )
    """

    if n_sampling < 3:
        raise ValueError('Cannot compute MoEs because n_sampling < 3.')

    if ceil(0.632 * target.size) < 3:
        raise ValueError('Cannot compute MoEs because 0.632 * n_sample < 3.')

    print('Computing MoEs with {} samplings ...'.format(n_sampling))
    feature_x_sampling = empty((features.shape[0], n_sampling))

    seed(random_seed)
    for i in range(n_sampling):
        if i % (n_sampling // 3) == 0:
            print('\t{}/{} ...'.format(i + 1, n_sampling))

        random_indices = choice(target.size, ceil(0.632 * target.size))
        sampled_target = target[random_indices]
        sampled_features = features[:, random_indices]

        random_state = get_state()

        feature_x_sampling[:, i] = match_target_and_features(
            sampled_target, sampled_features, min_n_sample, match_function)

        set_state(random_state)
    print('\t{}/{} - done.'.format(i + 1, n_sampling))

    return apply_along_axis(compute_margin_of_error, 1, feature_x_sampling)


def permute_target_and_match_target_and_features(target, features,
                                                 min_n_sample, match_function,
                                                 n_permutation, random_seed):
    """
    Permute target and match target and features.
    Arguments:
        target (ndarray): (n_sample, )
        features (ndarray): (n_feature, n_sample, )
        min_n_sample (int):
        match_function (callable):
        n_permutation (int): 1 <= n_permutation
        random_seed (float):
    Returns:
        ndarray: (n_feature, n_permutation, )
    """

    if n_permutation < 1:
        raise ValueError(
            'Not computing P-Value and FDR because n_permutation < 1.')

    print('Computing p-values and FDRs with {} permutations ...'.format(
        n_permutation))

    feature_x_permutation = empty((features.shape[0], n_permutation))

    permuted_target = array(target)

    seed(random_seed)
    for i in range(n_permutation):
        if i % (n_permutation // 3) == 0:
            print('\t{}/{} ...'.format(i + 1, n_permutation))

        shuffle(permuted_target)

        random_state = get_state()

        feature_x_permutation[:, i] = match_target_and_features(
            permuted_target, features, min_n_sample, match_function)

        set_state(random_state)
    print('\t{}/{} - done.'.format(i + 1, n_permutation))

    return feature_x_permutation


def match_target_and_features(target, features, min_n_sample, match_function):
    """
    Drop nan from target and features[i] and compute: scores[i] = match_function(
    target, features[i]).
    Arguments:
        target (ndarray): (n_sample, )
        features (ndarray): (n_feature, n_sample, )
        min_n_sample (int):
        match_function (callable):
    Returns:
        ndarray: (n_feature, )
    """

    return apply_along_axis(drop_nan_and_apply_function_on_2_1d_arrays, 1,
                            features, target, min_n_sample, match_function)
