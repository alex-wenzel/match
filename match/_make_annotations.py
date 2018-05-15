from pandas import DataFrame


def _make_annotations(scores):

    annotations = DataFrame(index=scores.index)

    if scores['0.95 MoE'].isna().all():

        annotations['IC'] = scores['Score'].apply('{:.2f}'.format)

    else:

        annotations['IC(\u0394)'] = scores[['Score', '0.95 MoE']].apply(
            lambda score_moe: '{:.2f}({:.2f})'.format(*score_moe), axis=1)

    if not scores['P-Value'].isna().all():

        function = '{:.2e}'.format

        annotations['P-Value'] = scores['P-Value'].apply(function)

        annotations['FDR'] = scores['FDR'].apply(function)

    return annotations
