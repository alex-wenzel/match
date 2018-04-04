from warnings import warn

from numpy import nan_to_num
from pandas import DataFrame

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


def make_summary_match_panel(
        target,
        multiple_features,
        plot_only_columns_shared_by_target_and_all_features=False,
        target_ascending=False,
        cluster_within_category=True,
        min_n_sample=5,
        match_function=compute_information_coefficient,
        random_seed=20121020,
        n_sampling=0,
        n_permutation=0,
        target_type='continuous',
        plot_max_std=3,
        title='Summary Match Panel',
        html_file_path=None):

    if plot_only_columns_shared_by_target_and_all_features:
        for name, d in multiple_features.items():
            target = target.loc[target.index & d['df'].columns]

    if isinstance(target_ascending, bool):
        target.sort_values(ascending=target_ascending, inplace=True)

    elif cluster_within_category and not nd_array_is_sorted(target.values):
        cluster_within_category = False
        warn(
            'Set cluster_within_category=False because target is not increasing or decreasing.'
        )

    target, target_min, target_max, target_colorscale = process_target_or_features_for_plotting(
        target, target_type, plot_max_std)
    target_df = target.to_frame().T

    layout = dict(
        width=800,
        margin=dict(l=160, r=160),
        title=title,
        xaxis1=dict(anchor='y1'))

    data = []
    layout_annotations = []

    n_row = 1
    for i, (name, d) in enumerate(multiple_features.items()):
        n_row += 1
        n_row += len(d['indices'])

    layout.update(height=max(800, n_row * 80))
    row_fraction = 1 / n_row

    yaxis_name = 'yaxis{}'.format(len(multiple_features) + 1)
    domain_end = 1
    domain_start = domain_end - row_fraction
    layout[yaxis_name] = dict(domain=(domain_start, domain_end))
    domain_end = domain_start - row_fraction

    data.append(
        dict(
            type='heatmap',
            showlegend=True,
            yaxis=yaxis_name.replace('axis', ''),
            z=target_df.values[::-1],
            x=target_df.columns,
            y=target_df.index[::-1],
            colorscale=target_colorscale,
            showscale=False,
            zmin=target_min,
            zmax=target_max))

    for i, (name, d) in enumerate(multiple_features.items()):
        print('Making match panel for {} ...'.format(name))

        features = d['df']
        indices = d['indices']
        index_aliases = d['index_aliases']
        emphasis = d['emphasis']
        data_type = d['data_type']

        missing_indices = tuple(
            index for index in indices if index not in features.index)
        if len(missing_indices):
            raise ValueError(
                'features do not have indices {}.'.format(missing_indices))
        features = features.loc[indices]

        features = drop_df_slices(
            features.reindex(columns=target.index), 1, max_n_unique_object=1)

        if cluster_within_category and target_type in ('binary',
                                                       'categorical'):
            if all(1 < (target == value).sum() for value in target):
                features = features.iloc[:,
                                         cluster_2d_array_slices_by_group(
                                             nan_to_num(features.values),
                                             nan_to_num(target.values))]

        scores = match(
            target.values,
            features.values,
            min_n_sample,
            match_function,
            n_top_feature=features.shape[0],
            n_sampling=n_sampling,
            n_permutation=n_permutation,
            random_seed=random_seed)
        scores.index = features.index

        scores = scores.sort_values('Score', ascending=emphasis == 'low')
        features = features.loc[scores.index]

        features.index = features.index.map(
            {index: alias
             for index, alias in zip(indices, index_aliases)}.get)

        annotations = DataFrame(index=scores.index)

        if scores['0.95 MoE'].isna().all():
            annotations['IC'] = [
                '{:.2f}'.format(score) for score in scores['Score']
            ]
        else:
            annotations['IC(\u0394)'] = scores[[
                'Score',
                '0.95 MoE'
            ]].apply(
                lambda score_margin_of_error: '{:.2f}({:.2f})'.format(*score_margin_of_error), axis=1)

        if not scores['P-Value'].isna().all():
            annotations['P-Value'] = scores['P-Value'].apply('{:.2e}'.format)
            annotations['FDR'] = scores['FDR'].apply('{:.2e}'.format)

        features, features_min, features_max, features_colorscale = process_target_or_features_for_plotting(
            features, data_type, plot_max_std)

        yaxis_name = 'yaxis{}'.format(len(multiple_features) - i)
        domain_start = domain_end - len(d['indices']) * row_fraction
        layout[yaxis_name] = dict(domain=(domain_start, domain_end))
        domain_end = domain_start - row_fraction

        data.append(
            dict(
                type='heatmap',
                yaxis=yaxis_name.replace('axis', ''),
                z=features.values[::-1],
                x=features.columns,
                y=features.index[::-1],
                colorscale=features_colorscale,
                showscale=False,
                zmin=features_min,
                zmax=features_max))

        for i, (column_name, annotation_column) in enumerate(
                annotations.items()):
            x = 1.05 + i / 10
            y = 1 - (row_fraction / 2)

            layout_annotations.append(
                dict(
                    xref='paper',
                    yref='paper',
                    x=x,
                    y=y,
                    xanchor='center',
                    yanchor='middle',
                    text='{}'.format(column_name),
                    showarrow=False))

            y = domain_end - row_fraction / 2

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
                        text='{:.3f}'.format(y),
                        showarrow=False))

    layout.update(annotations=layout_annotations)

    figure = dict(data=data, layout=layout)

    plot_and_save(figure, html_file_path)

    return figure
