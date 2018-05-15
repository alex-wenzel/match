def _check_features_index(features):

    if any(str(i).isdigit() for i in features.index):

        raise ValueError('Use only non-digit features index.')
