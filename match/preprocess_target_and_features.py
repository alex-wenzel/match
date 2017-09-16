from pandas import Series

from .support.support.df import drop_slices


def preprocess_target_and_features(target,
                                   features,
                                   target_ascending,
                                   max_n_unique_objects_for_drop_slices,
                                   indexs=()):
    """
    Make sure target is a Series. Keep only shared columns. Drop rows with less
        than max_n_unique_objects unique values.
    Arguments:
        target (iterable | Series):
        features (DataFrame):
        indexs (iterable):
        target_ascending (bool):
        max_n_unique_objects_for_drop_slices (int):
        indexs (iterable): target indexs and features columns to keep
    Returns:
        Series: Target
        DataFrame: Features
    """

    if not isinstance(target, Series):
        target = Series(target, index=features.columns)

    if not len(indexs):
        indexs = target.index & features.columns
        print('Target {} {} and features {} have {} shared columns.'.format(
            target.name, target.shape, features.shape, len(indexs)))

        if not len(indexs):
            raise ValueError(
                'Target {} {} and features {} have {} shared columns.'.format(
                    target.name, target.shape, features.shape, len(indexs)))

    target = target[indexs]
    target.sort_values(ascending=target_ascending, inplace=True)

    features = features[target.index]

    features = drop_slices(
        features,
        max_n_unique_objects=max_n_unique_objects_for_drop_slices,
        axis=1)

    if features.empty:
        raise ValueError('No feature has at least {} unique objects.'.format(
            max_n_unique_objects_for_drop_slices))

    return target, features
