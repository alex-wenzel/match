from numpy import finfo
from pandas import concat

from ._check_features_index import _check_features_index
from ._make_annotations import _make_annotations
from ._match import _match
from ._process_target_or_features_for_plotting import \
    _process_target_or_features_for_plotting
from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .plot.plot.plot_and_save import plot_and_save
from .support.support.iterable import make_object_int_mapping
from .support.support.df import drop_df_slice

EPS = finfo(float).eps


def make_summary_match_panel(
        target,
        feature_dicts,
        target_ascending=False,
        target_type='continuous',
        plot_target_std_max=None,
        plot_only_columns_shared_by_target_and_all_features=False,
        match_function=compute_information_coefficient,
        n_required_for_match_function=2,
        raise_for_n_less_than_required=False,
        random_seed=20121020,
        n_sampling=None,
        n_permutation=None,
        plot_features_std_max=None,
        title='Summary Match Panel',
        layout_width=960,
        row_height=64,
        layout_side_margin=200,
        annotation_font_size=9.6,
        html_file_path=None,
        plotly_file_path=None,
):

    if target.dtype == 'O':

        target = target.map(make_object_int_mapping(target)[0])

    if isinstance(
            target_ascending,
            bool,
    ):

        target.sort_values(
            ascending=target_ascending,
            inplace=True,
        )

    target, target_plot_min, target_plot_max, target_colorscale = _process_target_or_features_for_plotting(
        target,
        target_type,
        plot_target_std_max,
    )

    target_df = target.to_frame().T

    if plot_only_columns_shared_by_target_and_all_features:

        for features_dict in feature_dicts.values():

            target = target.loc[target.index & features_dict['df'].columns]

    n_row = 1 + len(feature_dicts)

    for features_dict in feature_dicts.values():

        n_row += len(features_dict['indices'])

    layout = dict(
        width=layout_width,
        margin=dict(
            l=layout_side_margin,
            r=layout_side_margin,
        ),
        xaxis=dict(anchor='y'),
        height=row_height / 2 * max(
            8,
            n_row,
        ),
        title=title,
        annotations=[],
    )

    row_fraction = 1 / n_row

    yaxis_name = 'yaxis{}'.format(len(feature_dicts) + 1).replace(
        'axis1',
        'axis',
    )

    domain_end = 1

    domain_start = domain_end - row_fraction

    if abs(domain_start) < EPS:

        domain_start = 0

    layout[yaxis_name] = dict(
        domain=(
            domain_start,
            domain_end,
        ),
        tickfont=dict(size=annotation_font_size),
    )

    data = [
        dict(
            yaxis=yaxis_name.replace(
                'axis',
                '',
            ),
            type='heatmap',
            z=target_df.values[::-1],
            x=target_df.columns,
            y=target_df.index[::-1],
            text=(target_df.columns, ),
            zmin=target_plot_min,
            zmax=target_plot_max,
            colorscale=target_colorscale,
            showscale=False,
        )
    ]

    multiple_scores = []

    for features_index, (
            name,
            features_dict,
    ) in enumerate(feature_dicts.items()):

        print('Making match panel for {} ...'.format(name))

        df = features_dict['df']

        indices = list(features_dict['indices'])

        features = df.loc[indices, target.index]

        _check_features_index(features)

        features = drop_df_slice(
            features,
            1,
            min_n_not_na_unique_value=2,
        )

        if features.empty:

            continue

        if 'score' in features_dict:

            scores = features_dict['score'].loc[indices]

        else:

            scores = _match(
                target.values,
                features.values,
                1,
                match_function,
                n_required_for_match_function,
                raise_for_n_less_than_required,
                None,
                random_seed,
                n_sampling,
                n_permutation,
            )

            scores.index = features.index

        scores.sort_values(
            'Score',
            ascending=features_dict['emphasis'] == 'low',
            inplace=True,
        )

        multiple_scores.append(scores)

        features_to_plot = features.loc[scores.index]

        if 'index_aliases' in features_dict:

            features_to_plot.index = features_to_plot.index.map({
                index: alias
                for index, alias in zip(
                    features.index,
                    features_dict['index_aliases'],
                )
            }.get)

        annotations = _make_annotations(scores.dropna(
            axis=1,
            how='all',
        ))

        features_to_plot, features_plot_min, features_plot_max, features_colorscale = _process_target_or_features_for_plotting(
            features_to_plot,
            features_dict['data_type'],
            plot_features_std_max,
        )

        yaxis_name = 'yaxis{}'.format(len(feature_dicts) -
                                      features_index).replace(
                                          'axis1',
                                          'axis',
                                      )

        domain_end = domain_start - row_fraction

        if abs(domain_end) < EPS:

            domain_end = 0

        domain_start = domain_end - len(
            features_dict['indices']) * row_fraction

        if abs(domain_start) < EPS:

            domain_start = 0

        layout[yaxis_name] = dict(
            domain=(
                domain_start,
                domain_end,
            ),
            dtick=1,
            tickfont=dict(size=annotation_font_size),
        )

        data.append(
            dict(
                yaxis=yaxis_name.replace(
                    'axis',
                    '',
                ),
                type='heatmap',
                z=features_to_plot.values[::-1],
                x=features_to_plot.columns,
                y=features_to_plot.index[::-1],
                zmin=features_plot_min,
                zmax=features_plot_max,
                colorscale=features_colorscale,
                showscale=False,
            ))

        layout_annotation_template = dict(
            xref='paper',
            yref='paper',
            xanchor='left',
            yanchor='middle',
            font=dict(size=annotation_font_size),
            width=64,
            showarrow=False,
        )

        for annotation_index, (annotation,
                               strs) in enumerate(annotations.items()):

            x = 1.0016 + annotation_index / 10

            if features_index == 0:

                layout['annotations'].append(
                    dict(
                        x=x,
                        y=1 - (row_fraction / 2),
                        text='<b>{}</b>'.format(annotation),
                        **layout_annotation_template,
                    ))

            y = domain_end - (row_fraction / 2)

            for str_ in strs:

                layout['annotations'].append(
                    dict(
                        x=x,
                        y=y,
                        text='<b>{}</b>'.format(str_),
                        **layout_annotation_template,
                    ))

                y -= row_fraction

    plot_and_save(
        dict(
            layout=layout,
            data=data,
        ),
        html_file_path,
        plotly_file_path,
    )

    return concat(multiple_scores).sort_values('Score')
