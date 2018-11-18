from warnings import warn

from numpy import nan_to_num

from ._make_annotations import _make_annotations
from ._match import _match
from ._process_target_or_features_for_plotting import \
    _process_target_or_features_for_plotting
from ._style import (ANNOTATION_FONT_SIZE, LAYOUT_SIDE_MARGIN, LAYOUT_WIDTH,
                     ROW_HEIGHT)
from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .nd_array.nd_array.cluster_2d_array_slices_by_group import \
    cluster_2d_array_slices_by_group
from .nd_array.nd_array.nd_array_is_sorted import nd_array_is_sorted
from .plot.plot.plot_and_save import plot_and_save
from .support.support.iterable import make_object_int_mapping
from .support.support.path import (combine_path_prefix_and_suffix,
                                   establish_path)
from .support.support.series import get_extreme_series_indices


def make_match_panel(
        target,
        features,
        target_ascending=True,
        score_moe_p_value_fdr=None,
        n_job=1,
        match_function=compute_information_coefficient,
        n_required_for_match_function=2,
        raise_for_n_less_than_required=False,
        extreme_feature_threshold=8,
        random_seed=20121020,
        n_sampling=0,
        n_permutation=0,
        target_type='continuous',
        cluster_within_category=True,
        features_type='continuous',
        plot_std_max=None,
        title='Match Panel',
        layout_width=LAYOUT_WIDTH,
        row_height=ROW_HEIGHT,
        layout_side_margin=LAYOUT_SIDE_MARGIN,
        annotation_font_size=ANNOTATION_FONT_SIZE,
        file_path_prefix=None,
        plotly_file_path_prefix=None,
):

    common_indices = target.index & features.columns

    print(
        'target.index ({}) & features.columns ({}) have {} in common.'.format(
            target.index.size,
            features.columns.size,
            len(common_indices),
        ))

    target = target[common_indices]

    if target.dtype == 'O':

        target = target.map(make_object_int_mapping(target)[0])

    if target_ascending is not None:

        target.sort_values(
            ascending=target_ascending,
            inplace=True,
        )

    features = features[target.index]

    if file_path_prefix is not None:

        establish_path(
            file_path_prefix,
            'file',
        )

    if score_moe_p_value_fdr is None:

        score_moe_p_value_fdr = _match(
            target.values,
            features.values,
            n_job,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
            extreme_feature_threshold,
            random_seed,
            n_sampling,
            n_permutation,
        )

        score_moe_p_value_fdr.index = features.index

        if file_path_prefix is not None:

            score_moe_p_value_fdr.to_csv(
                '{}.tsv'.format(file_path_prefix),
                sep='\t',
            )

    indices = get_extreme_series_indices(
        score_moe_p_value_fdr['Score'],
        extreme_feature_threshold,
        ascending=False,
    )

    if not len(indices):

        return score_moe_p_value_fdr

    features_to_plot = features.loc[indices]

    scores_to_plot = score_moe_p_value_fdr.loc[indices]

    annotations = _make_annotations(
        scores_to_plot.dropna(
            axis=1,
            how='all',
        ))

    target, target_plot_min, target_plot_max, target_colorscale = _process_target_or_features_for_plotting(
        target,
        target_type,
        plot_std_max,
    )

    if target_type in (
            'binary',
            'categorical',
    ) and cluster_within_category:

        if target.value_counts().min() < 2:

            warn('Not clustering because a category has less than 2 values.')

        elif not nd_array_is_sorted(target.values):

            warn('Not clustering because target is not sorted.')

        else:

            clustered_indices = cluster_2d_array_slices_by_group(
                nan_to_num(features_to_plot.values),
                nan_to_num(target.values),
                1,
            )

            target = target.iloc[clustered_indices]

            features_to_plot = features_to_plot.iloc[:, clustered_indices]

    features_to_plot, features_plot_min, features_plot_max, features_colorscale = _process_target_or_features_for_plotting(
        features_to_plot,
        features_type,
        plot_std_max,
    )

    target_row_fraction = max(
        0.01,
        1 / (features_to_plot.shape[0] + 2),
    )

    target_yaxis_domain = (
        1 - target_row_fraction,
        1,
    )

    features_yaxis_domain = (
        0,
        1 - target_row_fraction * 2,
    )

    feature_row_fraction = (features_yaxis_domain[1] - features_yaxis_domain[0]
                            ) / features_to_plot.shape[0]

    layout = dict(
        width=layout_width,
        height=row_height * max(
            10,
            (features_to_plot.shape[0] + 2)**0.8,
        ),
        margin=dict(
            l=layout_side_margin,
            r=layout_side_margin,
        ),
        xaxis=dict(
            anchor='y',
            tickfont=dict(size=annotation_font_size),
        ),
        yaxis=dict(
            domain=features_yaxis_domain,
            dtick=1,
            tickfont=dict(size=annotation_font_size),
        ),
        yaxis2=dict(
            domain=target_yaxis_domain,
            tickfont=dict(size=annotation_font_size),
        ),
        title=title,
        annotations=[],
    )

    data = [
        dict(
            yaxis='y2',
            type='heatmap',
            z=target.to_frame().T.values,
            x=target.index,
            y=(target.name, ),
            text=(target.index, ),
            zmin=-1,
            zmax=1,
            colorscale=target_colorscale,
            showscale=False,
        ),
        dict(
            yaxis='y',
            type='heatmap',
            z=features_to_plot.values[::-1],
            x=features_to_plot.columns,
            y=features_to_plot.index[::-1],
            zmin=-1,
            zmax=1,
            colorscale=features_colorscale,
            showscale=False,
        ),
    ]

    layout_annotation_template = dict(
        xref='paper',
        yref='paper',
        xanchor='left',
        yanchor='middle',
        font=dict(size=annotation_font_size),
        width=64,
        showarrow=False,
    )

    for annotation_index, (
            annotation,
            strs,
    ) in enumerate(annotations.items()):

        x = 1.0016 + annotation_index / 10

        layout['annotations'].append(
            dict(
                x=x,
                y=target_yaxis_domain[1] - (target_row_fraction / 2),
                text='<b>{}</b>'.format(annotation),
                **layout_annotation_template,
            ))

        y = features_yaxis_domain[1] - (feature_row_fraction / 2)

        for str_ in strs:

            layout['annotations'].append(
                dict(
                    x=x,
                    y=y,
                    text='<b>{}</b>'.format(str_),
                    **layout_annotation_template,
                ))

            y -= feature_row_fraction

    suffix = '.html'

    plot_and_save(
        dict(
            layout=layout,
            data=data,
        ),
        combine_path_prefix_and_suffix(
            file_path_prefix,
            suffix,
            False,
        ),
        combine_path_prefix_and_suffix(
            plotly_file_path_prefix,
            suffix,
            False,
        ),
    )

    return score_moe_p_value_fdr
