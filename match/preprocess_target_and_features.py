from pandas import Series

from .support.support.df import drop_slices


def _preprocess_target_and_features(
        target, features, keep_only_target_columns_with_value,
        target_ascending, max_n_unique_objects_for_drop_slices):
    """
    Make target Series. Select columns. Drop rows with less than
        max_n_unique_objects unique values.
    Arguments:
        target (iterable | Series):
        features (DataFrame):
        keep_only_target_columns_with_value (bool):
        target_ascending (bool):
        max_n_unique_objects_for_drop_slices (int):
    Returns:
        Series: Target
        DataFrame: Features
    """

    # Make target Series, assuming ordered
    if not isinstance(target, Series):
        target = Series(target, index=features.columns)

    # Select columns
    i = target.index & features.columns
    print('Target {} {} and features {} have {} shared columns.'.format(
        target.name, target.shape, features.shape, len(i)))

    if not len(i):
        raise ValueError('0 column.')

    target = target[i]
    target.sort_values(ascending=target_ascending, inplace=True)

    features = features[target.index]

    # Drop rows with less than max_n_unique_objects unique values
    features = drop_slices(
        features,
        max_n_unique_objects=max_n_unique_objects_for_drop_slices,
        axis=1)

    if features.empty:
        raise ValueError('No feature has at least {} unique objects.'.format(
            max_n_unique_objects_for_drop_slices))

    return target, features
