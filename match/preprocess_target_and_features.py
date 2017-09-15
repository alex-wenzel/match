from pandas import Series

from .support.support.df import drop_slices


def _preprocess_target_and_features(
        target, features, keep_only_target_columns_with_value,
        target_ascending, max_n_unique_objects_for_drop_slices):
    """
    Make target Series. Select columns. Drop rows with less than
        max_n_unique_objects unique values.
    :param target: iterable | Series
    :param features: DataFrame
    :param keep_only_target_columns_with_value: bool
    :param target_ascending: bool
    :param max_n_unique_objects_for_drop_slices: int
    :return: Series & DataFrame
    """

    # Make target Series
    if not isinstance(target, Series):
        target = Series(target, index=features.columns)

    # Select columns
    if keep_only_target_columns_with_value:
        i = target.index & features.columns
        print('Target {} {} and features {} have {} shared columns.'.format(
            target.name, target.shape, features.shape, len(i)))
    else:
        i = target.index

    if not len(i):
        raise ValueError('0 column.')

    target = target[i]
    target.sort_values(ascending=target_ascending, inplace=True)

    features = features.loc[:, target.index]

    # Drop rows with less than max_n_unique_objects unique values
    features = drop_slices(
        features,
        max_n_unique_objects=max_n_unique_objects_for_drop_slices,
        axis=1)

    if features.empty:
        raise ValueError('No feature has at least {} unique objects.'.format(
            max_n_unique_objects_for_drop_slices))

    return target, features
