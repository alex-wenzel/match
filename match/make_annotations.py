from pandas import DataFrame


def make_annotations(scores):

    annotations = DataFrame(index=scores.index)

    if scores['0.95 MoE'].isna().all():
        annotations['IC'] = scores['Score'].apply('{:.2f}'.format)
    else:
        annotations['IC(\u0394)'] = scores[['Score', '0.95 MoE']].apply(
            lambda score_moe: '{:.2f}({:.2f})'.format(*score_moe), axis=1)

    if not scores['P-Value'].isna().all():
        annotations['P-Value'] = scores['P-Value'].apply('{:.2e}'.format)
        annotations['FDR'] = scores['FDR'].apply('{:.2e}'.format)

    return annotations
