
    features = drop_slices(
        features,
        max_n_unique_objects=max_n_unique_objects_for_drop_slices,
        axis=1)

    if features.empty:
        raise ValueError('No feature has at least {} unique objects.'.format(
            max_n_unique_objects_for_drop_slices))

    return target, features
