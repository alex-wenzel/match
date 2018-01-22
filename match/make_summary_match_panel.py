from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from pandas import DataFrame

from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .match import match
from .nd_array.nd_array.cluster_2d_array_slices_by_group import \
    cluster_2d_array_slices_by_group
from .plot.plot.save_plot import save_plot
from .plot.plot.style import FIGURE_SIZE, FONT_LARGER, FONT_LARGEST
from .plot_match_panel import plot_match_panel
from .support.support.df import drop_df_slices

RANDOM_SEED = 20121020


def make_summary_match_panel(
        target,
        multiple_features,
        plot_only_columns_shared_by_target_and_all_features=False,
        target_ascending=False,
        min_n_sample=5,
        function_=compute_information_coefficient,
        random_seed=RANDOM_SEED,
        n_sampling=8,
        n_permutation=8,
        target_type='continuous',
        max_std=3,
        title='Summary Match Panel',
        target_int_to_str=None,
        target_annotation_kwargs=None,
        max_ytick_size=26,
        plot_column_names=False,
        file_path=None):
    """
    Make summary match panel.
    Arguments:
        target (Series): (n_sample, )
        multiple_features (dict):
            {
                name :
                    {
                        df,
                        indices,
                        index_aliases,
                        emphasis,
                        data_type,
                    },
                ...,
            }
        plot_only_columns_shared_by_target_and_all_features (bool):
        target_ascending (bool): True if target increase from left to right |
            False right to left
        min_n_sample (int):
        function_ (callable): function for computing match scores between the
            target and each feature
        random_seed (float):
        n_sampling (int): number of bootstrap samplings to build distribution
            to compute MoE; 3 <= n_sampling
        n_permutation (int): number of permutations for permutation test to
            compute P-Value and FDR
        target_type (str): 'continuous' | 'categorical' | 'binary'
        max_std (float):
        title (str): plot title
        target_int_to_str (dict):
            {
                int: str,
                ...,
            }
        target_annotation_kwargs (dict):
        max_ytick_size (int):
        plot_column_names (bool): whether to plot column names
        file_path (str):
    Returns:
    """

    target_annotation_kwargs_ = {
        'fontsize': 12,
    }
    if target_annotation_kwargs is not None:
        target_annotation_kwargs_.update(target_annotation_kwargs)
    target_annotation_kwargs = target_annotation_kwargs_

    n = 0
    max_width = 0
    for name, d in multiple_features.items():
        n += len(d['indices']) + 3
        w = d['df'].shape[1]
        if max_width < w:
            max_width = w

    fig = figure(figsize=(min(pow(max_width, 1.8), FIGURE_SIZE[1]), n))

    gridspec = GridSpec(n, 1)

    fig.text(
        0.5,
        0.88,
        title,
        horizontalalignment='center',
        verticalalignment='bottom',
        **FONT_LARGEST)
    r_i = 0

    columns = target.index
    if plot_only_columns_shared_by_target_and_all_features:
        for name, d in multiple_features.items():
            columns &= d['df'].columns

    for fi, (
            name,
            d, ) in enumerate(multiple_features.items()):
        print('Making match panel for {} ...'.format(name))

        features = d['df']
        indices = d['indices']
        index_aliases = d['index_aliases']
        emphasis = d['emphasis']
        data_type = d['data_type']

        missing_indices = (i for i in indices if i not in features.index)
        if any(missing_indices):
            raise ValueError(
                'features don\'t have indices {}.'.format(missing_indices))

        target = target.loc[columns & features.columns].sort_values(
            ascending=target_ascending)
        features = features[target.index]

        features = drop_df_slices(
            features.loc[indices], 1, max_n_unique_object=1)

        if target_type in (
                'binary',
                'categorical', ):
            features = features.iloc[:,
                                     cluster_2d_array_slices_by_group(
                                         features.values, target.values)]

        scores = match(
            target.values,
            features.values,
            min_n_sample,
            function_,
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
        annotations['IC(\u0394)'] = scores[[
            'Score',
            '0.95 MoE',
        ]].apply(
            lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)
        annotations['P-Value'] = scores['P-Value'].apply('{:.2e}'.format)
        annotations['FDR'] = scores['FDR'].apply('{:.2e}'.format)

        title_ax = subplot(gridspec[r_i:r_i + 1, 0])
        r_i += 1
        title_ax.axis('off')
        title_ax.text(
            0.5,
            0,
            '{} (n={})'.format(name, target.size),
            horizontalalignment='center',
            **FONT_LARGER)

        target_ax = subplot(gridspec[r_i:r_i + 1, 0])
        r_i += 1

        features_ax = subplot(gridspec[r_i:r_i + features.shape[0], 0])
        r_i += features.shape[0]

        plot_match_panel(target, features, target_type, data_type, max_std,
                         target_ax, features_ax, None, target_int_to_str,
                         target_annotation_kwargs, max_ytick_size, annotations,
                         plot_column_names
                         and fi == len(multiple_features) - 1, None)

    if file_path:
        save_plot(file_path)
