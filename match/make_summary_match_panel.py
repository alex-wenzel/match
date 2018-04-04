from warnings import warn

from numpy import nan_to_num

from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .make_annotations import make_annotations
from .make_match_panel import (LAYOUT_ANNOTATION_TEMPLATE,
                               MATCH_PANEL_LAYOUT_TEMPLATE)
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
        features_dict,
        plot_only_columns_shared_by_target_and_all_features=False,
        target_ascending=False,
        cluster_within_category=True,
        min_n_sample=2,
        match_function=compute_information_coefficient,
        random_seed=20121020,
        n_sampling=0,
        n_permutation=0,
        target_type='continuous',
        plot_max_std=3,
        title='Summary Match Panel',
        html_file_path=None):

    if plot_only_columns_shared_by_target_and_all_features:
        for name, features_dict in features_dict.items():
            target = target.loc[target.index & features_dict['df'].columns]

    if isinstance(target_ascending, bool):
        target.sort_values(ascending=target_ascending, inplace=True)

    elif cluster_within_category and not nd_array_is_sorted(target.values):
        cluster_within_category = False
        warn('Set cluster_within_category=False because target is not sorted.')

    target, target_min, target_max, target_colorscale = process_target_or_features_for_plotting(
        target, target_type, plot_max_std)
    target_df = target.to_frame().T

    layout = MATCH_PANEL_LAYOUT_TEMPLATE

    n_row = 1
    for i, (name, features_dict) in enumerate(features_dict.items()):
        n_row += 1
        n_row += len(features_dict['indices'])
    row_fraction = 1 / n_row

    layout.update(title=title, height=max(layout['height'], n_row * 24))

    data = []
    layout_annotations = []

    yaxis_name = 'yaxis{}'.format(len(features_dict) + 1)
    domain_end = 1
    domain_start = domain_end - row_fraction
    layout[yaxis_name] = dict(
        domain=(domain_start, domain_end), ticks='', showticklabels=False)

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

    for i, (name, features_dict) in enumerate(features_dict.items()):
        print('Making match panel for {} ...'.format(name))

        features = features_dict['df']
        indices = features_dict['indices']
        index_aliases = features_dict['index_aliases']
        emphasis = features_dict['emphasis']
        data_type = features_dict['data_type']

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
        scores.sort_values('Score', ascending=emphasis == 'low', inplace=True)

        features = features.loc[scores.index]
        features.index = features.index.map(
            {index: alias
             for index, alias in zip(indices, index_aliases)}.get)

        annotations = make_annotations(scores)

        features, features_min, features_max, features_colorscale = process_target_or_features_for_plotting(
            features, data_type, plot_max_std)

        yaxis_name = 'yaxis{}'.format(len(features_dict) - i)

        domain_end = domain_start - row_fraction
        domain_start = domain_end - len(
            features_dict['indices']) * row_fraction
        layout[yaxis_name] = dict(domain=(domain_start, domain_end), dtick=1)

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

        for j, (annotation, strs) in enumerate(annotations.items()):
            x = 1.07 + i / 7

            if j == 0:
                y = 1 - (row_fraction / 2)
                layout_annotations.append(
                    dict(
                        x=x,
                        y=y,
                        text=annotation,
                        **LAYOUT_ANNOTATION_TEMPLATE))

            y = domain_end - (row_fraction / 2)

            for str_ in strs:
                layout_annotations.append(
                    dict(x=x, y=y, text=str_, **LAYOUT_ANNOTATION_TEMPLATE))
                y -= row_fraction

    layout.update(annotations=layout_annotations)

    figure = dict(data=data, layout=layout)

    plot_and_save(figure, html_file_path)
