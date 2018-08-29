from .support.support.str_ import make_float_str


def _make_annotations(scores):

    annotations = scores.applymap(make_float_str)

    if '0.95 MoE' in annotations.columns:

        annotations['Score'] = tuple(
            _combine_score_str_and_moe_str(
                score_str,
                moe_str,
            ) for score_str, moe_str in zip(
                annotations['Score'],
                annotations.pop('0.95 MoE'),
            ))

    return annotations


def _combine_score_str_and_moe_str(
        score_str,
        moe_str,
):

    if moe_str == 'nan':

        return score_str

    else:

        return '{} \u00B1 {}'.format(
            score_str,
            moe_str,
        )
