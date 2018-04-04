from warnings import warn

from numpy import nan_to_num
from pandas import DataFrame, Index

from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .match import match
from .nd_array.nd_array.cluster_2d_array_slices_by_group import \
    cluster_2d_array_slices_by_group
from .nd_array.nd_array.nd_array_is_sorted import nd_array_is_sorted
from .plot.plot.plot_and_save import plot_and_save
from .process_target_or_features_for_plotting import \
    process_target_or_features_for_plotting
from .support.support.df import drop_df_slices
from .support.support.path import establish_path
from .support.support.series import get_top_and_bottom_series_indices


def make_match_panel(target,
                     features,
                     target_ascending=False,
                     cluster_within_category=True,
                     scores=None,
                     min_n_sample=5,
                     match_function=compute_information_coefficient,
                     random_seed=20121020,
                     n_job=1,
                     scores_ascending=False,
                     indices=None,
                     n_top_feature=10,
                     max_n_feature=100,
                     n_sampling=0,
                     n_permutation=0,
                     target_type='continuous',
                     features_type='continuous',
                     plot_max_std=3,
                     title='Match Panel',
                     target_xticklabels=(),
                     max_ytick_size=50,
                     plot_column_names=False,
                     file_path_prefix=None):

    target = target.loc[target.index & features.columns]

    if isinstance(target_ascending, bool):
        target.sort_values(ascending=target_ascending, inplace=True)

    elif cluster_within_category and not nd_array_is_sorted(target.values):
        cluster_within_category = False
        warn(
            'Set cluster_within_category=False because target is not increasing or decreasing.'
        )

    features = drop_df_slices(features[target.index], 1, max_n_unique_object=1)

    if cluster_within_category and target_type in ('binary', 'categorical'):
        if all(1 < (target == value).sum() for value in target):
            features = features.iloc[:,
                                     cluster_2d_array_slices_by_group(
                                         nan_to_num(features.values),
                                         nan_to_num(target.values))]

    if scores is None:
        scores = match(
            target.values,
            features.values,
            min_n_sample,
            match_function,
            n_job=n_job,
            n_top_feature=n_top_feature,
            max_n_feature=max_n_feature,
            n_sampling=n_sampling,
            n_permutation=n_permutation,
            random_seed=random_seed)
        scores.index = features.index

        scores.sort_values('Score', inplace=True)

        if file_path_prefix:
            tsv_file_path = file_path_prefix + '.match.tsv'
            establish_path(tsv_file_path, 'file')
            scores.to_csv(tsv_file_path, sep='\t')

    if indices is None:
        indices = get_top_and_bottom_series_indices(scores['Score'],
                                                    n_top_feature).tolist()

    indices = Index(
        sorted(
            indices,
            key=lambda index: scores.loc[index, 'Score'],
            reverse=not scores_ascending))

    if max_n_feature and max_n_feature < indices.size:
        indices = indices[:max_n_feature // 2].append(
            indices[-max_n_feature // 2:])

    scores_to_plot = scores.loc[indices]

    annotations = DataFrame(index=scores_to_plot.index)

    if scores_to_plot['0.95 MoE'].isna().all():
        annotations['IC'] = [
            '{:.2f}'.format(score) for score in scores_to_plot['Score']
        ]
    else:
        annotations['IC(\u0394)'] = scores_to_plot[[
            'Score',
            '0.95 MoE'
        ]].apply(
            lambda score_margin_of_error: '{:.2f}({:.2f})'.format(*score_margin_of_error), axis=1)

    if not scores_to_plot['P-Value'].isna().all():
        annotations['P-Value'] = scores_to_plot['P-Value'].apply(
            '{:.2e}'.format)
        annotations['FDR'] = scores_to_plot['FDR'].apply('{:.2e}'.format)

    if file_path_prefix:
        html_file_path = file_path_prefix + '.match.html'
    else:
        html_file_path = None

    target, target_min, target_max, target_colorscale = process_target_or_features_for_plotting(
        target, target_type, plot_max_std)
    target_df = target.to_frame().T

    features, features_min, features_max, features_colorscale = process_target_or_features_for_plotting(
        features.loc[scores_to_plot.index], features_type, plot_max_std)

    n_row = 2 + features.shape[0]
    row_fraction = 1 / n_row
    layout = dict(
        width=880,
        height=max(800, n_row * 80),
        margin=dict(l=100, r=(100, 240)[3 < n_permutation]),
        title=title,
        xaxis1=dict(anchor='y1'),
        yaxis1=dict(domain=(0, 1 - 2 * row_fraction)),
        yaxis2=dict(domain=(1 - row_fraction, 1)))

    data = []

    data.append(
        dict(
            type='heatmap',
            yaxis='y2',
            z=target_df.values[::-1],
            x=target_df.columns,
            y=target_df.index[::-1],
            colorscale=target_colorscale,
            showscale=False,
            zmin=target_min,
            zmax=target_max))

    data.append(
        dict(
            type='heatmap',
            yaxis='y1',
            z=features.values[::-1],
            x=features.columns,
            y=features.index[::-1],
            colorscale=features_colorscale,
            showscale=False,
            zmin=features_min,
            zmax=features_max))

    layout_annotations = []

    for i, (column_name, annotation_column) in enumerate(annotations.items()):
        x = 1.07 + i / 7
        y = 1 - (row_fraction / 2)

        layout_annotations.append(
            dict(
                xref='paper',
                yref='paper',
                x=x,
                y=y,
                xanchor='center',
                yanchor='middle',
                text=column_name,
                showarrow=False))

        y -= row_fraction

        for annotation in annotation_column:
            y -= row_fraction

            layout_annotations.append(
                dict(
                    xref='paper',
                    yref='paper',
                    x=x,
                    y=y,
                    xanchor='center',
                    yanchor='middle',
                    text=annotation,
                    showarrow=False))

    layout.update(annotations=layout_annotations)

    figure = dict(data=data, layout=layout)

    plot_and_save(figure, html_file_path)

    return scores
