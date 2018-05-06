from warnings import warn

from numpy import nan, nan_to_num

from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .make_annotations import make_annotations
from .match import match
from .nd_array.nd_array.cluster_2d_array_slices_by_group import \
    cluster_2d_array_slices_by_group
from .nd_array.nd_array.nd_array_is_sorted import nd_array_is_sorted
from .plot.plot.plot_and_save import plot_and_save
from .process_target_or_features_for_plotting import \
    process_target_or_features_for_plotting
from .support.support.df import drop_df_slice
from .support.support.iterable import make_object_int_mapping
from .support.support.path import establish_path
from .support.support.series import get_extreme_series_indices

LAYOUT_SIDE_MARGIN = 208

MATCH_PANEL_LAYOUT_TEMPLATE = dict(
    width=960,
    margin=dict(l=LAYOUT_SIDE_MARGIN, r=LAYOUT_SIDE_MARGIN),
    xaxis=dict(anchor='y'))

ROW_HEIGHT = 64

ANNOTATION_FONT_SIZE = 9.6

LAYOUT_ANNOTATION_TEMPLATE = dict(
    xref='paper',
    yref='paper',
    xanchor='left',
    yanchor='middle',
    font=dict(size=ANNOTATION_FONT_SIZE),
    width=64,
    showarrow=False)


def make_match_panel(target,
                     features,
                     target_ascending=False,
                     cluster_within_category=True,
                     scores=None,
                     min_n_sample=2,
                     match_function=compute_information_coefficient,
                     random_seed=20121020,
                     n_job=1,
                     scores_ascending=False,
                     extreme_feature_threshold=24,
                     n_sampling=0,
                     n_permutation=0,
                     target_type='continuous',
                     features_type='continuous',
                     plot_std_max=3,
                     title='Match Panel',
                     file_path_prefix=None):

    common_indices = target.index & features.columns

    n_common = len(common_indices)

    message = 'target.index ({}) & features.columns ({}) have {} in common.'.format(
        target.index.size, features.columns.size, n_common)

    if 0 < n_common:

        print(message)

        target = target.loc[common_indices]

    else:

        raise ValueError(message)

    if target.dtype == 'O':

        target = target.map(make_object_int_mapping(target)[0])

    if isinstance(target_ascending, bool):

        target.sort_values(ascending=target_ascending, inplace=True)

    features = drop_df_slice(
        features[target.index],
        1,
        only_object=nan,
        max_n_not_na_unique_object=1)

    if file_path_prefix:

        establish_path(file_path_prefix, 'file')

    if scores is None:

        scores = match(
            target.values,
            features.values,
            min_n_sample,
            match_function,
            n_job=n_job,
            extreme_feature_threshold=extreme_feature_threshold,
            n_sampling=n_sampling,
            n_permutation=n_permutation,
            random_seed=random_seed)

        scores.index = features.index

        scores.sort_values('Score', ascending=scores_ascending, inplace=True)

        if file_path_prefix:

            scores.to_csv(file_path_prefix + '.match.tsv', sep='\t')

    indices = get_extreme_series_indices(
        scores['Score'], extreme_feature_threshold, scores_ascending)

    features_to_plot = features.loc[indices]

    scores_to_plot = scores.loc[indices]

    annotations = make_annotations(scores_to_plot)

    target, target_min, target_max, target_colorscale = process_target_or_features_for_plotting(
        target, target_type, plot_std_max)

    target_df = target.to_frame().T

    if target_type in ('binary', 'categorical') and cluster_within_category:

        if target.value_counts().min() < 2:

            warn('Not clustering because a category has less than 2 values.')

        elif not nd_array_is_sorted(target.values):

            warn('Not clustering because target is not sorted.')

        else:

            features_to_plot = features_to_plot.iloc[:,
                                                     cluster_2d_array_slices_by_group(
                                                         nan_to_num(
                                                             features_to_plot.
                                                             values),
                                                         nan_to_num(
                                                             target.values),
                                                         1)]

    features_to_plot, features_min, features_max, features_colorscale = process_target_or_features_for_plotting(
        features_to_plot, features_type, plot_std_max)

    layout = MATCH_PANEL_LAYOUT_TEMPLATE

    layout['xaxis'].update(tickfont=dict(size=ANNOTATION_FONT_SIZE))

    target_row_fraction = max(0.01, 1 / (features_to_plot.shape[0] + 2))

    target_yaxis_domain = (1 - target_row_fraction, 1)

    features_yaxis_domain = (0, 1 - target_row_fraction * 2)

    feature_row_fraction = (features_yaxis_domain[1] - features_yaxis_domain[0]
                            ) / features_to_plot.shape[0]

    layout.update(
        height=ROW_HEIGHT * max(8, ((features_to_plot.shape[0] + 2)**0.8)),
        title=title,
        yaxis=dict(
            domain=features_yaxis_domain,
            dtick=1,
            tickfont=dict(size=ANNOTATION_FONT_SIZE)),
        yaxis2=dict(
            domain=target_yaxis_domain,
            tickfont=dict(size=ANNOTATION_FONT_SIZE)))

    data = [
        dict(
            yaxis='y2',
            type='heatmap',
            z=target_df.values[::-1],
            x=target_df.columns,
            y=target_df.index[::-1],
            zmin=target_min,
            zmax=target_max,
            colorscale=target_colorscale,
            showscale=False),
        dict(
            yaxis='y',
            type='heatmap',
            z=features_to_plot.values[::-1],
            x=features_to_plot.columns,
            y=features_to_plot.index[::-1],
            zmin=features_min,
            zmax=features_max,
            colorscale=features_colorscale,
            showscale=False)
    ]

    layout_annotations = []

    for annotation_index, (annotation, strs) in enumerate(annotations.items()):

        x = 1.008 + annotation_index / 8

        layout_annotations.append(
            dict(
                x=x,
                y=target_yaxis_domain[1] - (target_row_fraction / 2),
                text='<b>{}</b>'.format(annotation),
                **LAYOUT_ANNOTATION_TEMPLATE))

        y = features_yaxis_domain[1] - ((feature_row_fraction) / 2)

        for str_ in strs:

            layout_annotations.append(
                dict(
                    x=x,
                    y=y,
                    text='<b>{}</b>'.format(str_),
                    **LAYOUT_ANNOTATION_TEMPLATE))

            y -= feature_row_fraction

    layout.update(annotations=layout_annotations)

    if file_path_prefix:

        html_file_path = file_path_prefix + '.match_panel.html'

    else:

        html_file_path = None

    plot_and_save(dict(layout=layout, data=data), html_file_path)

    return scores
