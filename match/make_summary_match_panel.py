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
        function=compute_information_coefficient,
        n_samplings=30,
        n_permutations=30,
        random_seed=RANDOM_SEED,
        title='Summary Match Panel',
        target_type='continuous',
        max_std=3,
        target_annotation_kwargs={'fontsize': 12},
        plot_sample_names=False,
        file_path=None,
        dpi=100):
    """
    Make summary match panel.
    Arguments:
        target (Series): (n_samples)
        multiple_features (dict): {
            name : {
                df,
                indices,
                index_aliases,
                emphasis,
                data_type,
                }
            }
        plot_only_columns_shared_by_target_and_all_features (bool):
        target_ascending (bool): True if target increase from left to right,
            and False right to left
        function (callable): function for computing match scores between the
            target and each feature
        n_samplings (int): number of bootstrap samplings to build distribution
            to compute MoE; 3 <= n_samplings
        n_permutations (int): number of permutations for permutation test to
            compute p-values and FDR
        random_seed (int | array):
        title (str): plot title
        target_type (str): 'continuous' | 'categorical' | 'binary'
        max_std (number):
        target_annotation_kwargs (dict):
        plot_sample_names (bool): whether to plot column names
        file_path (str):
        dpi (int):
    Returns:
        None
    """

    # Compute the number of rows needed for plotting
    n = 0
    max_width = 0
    for name, d in multiple_features.items():
        n += len(d['indices']) + 3
        w = d['df'].shape[1]
        if max_width < w:
            max_width = w

    # Set up figure
    fig = figure(figsize=(min(pow(max_width, 1.8), FIGURE_SIZE[1]), n))

    # Set up ax grids
    gridspec = GridSpec(n, 1)

    # Plot title
    fig.text(
        0.5,
        0.88,
        title,
        horizontalalignment='center',
        verticalalignment='bottom',
        **FONT_LARGEST)
    r_i = 0

    # Set columns to be plotted
    columns = target.index
    if plot_only_columns_shared_by_target_and_all_features:
        for name, d in multiple_features.items():
            columns &= d['df'].columns

    # Plot multiple_features
    for fi, (name, d) in enumerate(multiple_features.items()):
        print('Making match panel for {} ...'.format(name))

        features = d['df']
        indices = d['indices']
        index_aliases = d['index_aliases']
        emphasis = d['emphasis']
        data_type = d['data_type']

        # Extract specified indices from features
        missing_indices = [i for i in indices if i not in features.index]
        if any(missing_indices):
            raise ValueError(
                'features don\'t have indices {}.'.format(missing_indices))

        # Sort target and features.columns (based on target)
        target = target.loc[columns & features.columns].sort_values(
            ascending=target_ascending or target.dtype == 'O')
        features = features[target.index]

        # Drop constant rows
        features = drop_df_slices(
            features.loc[indices], 1, max_n_unique_objects=1)

        target_o_to_int = {}
        target_int_to_o = {}
        if target.dtype == 'O':
            # Make target numerical
            for i, o in enumerate(target.unique()):
                target_o_to_int[o] = i
                target_int_to_o[i] = o
            target = target.map(target_o_to_int)

        if target_type in ('binary', 'categorical'):
            # Cluster within categories
            columns = cluster_2d_array_slices_by_group(features.values,
                                                       target.values)
            features = features.iloc[:, columns]

        # Match
        scores = match(
            target.values,
            features.values,
            function,
            n_features=features.shape[0],
            n_samplings=n_samplings,
            n_permutations=n_permutations,
            random_seed=random_seed)
        scores.index = features.index

        # Sort scores
        scores = scores.sort_values('Score', ascending=emphasis == 'low')
        features = features.loc[scores.index]

        # Use alias
        i_to_a = {i: a for i, a in zip(indices, index_aliases)}
        features.index = features.index.map(lambda i: i_to_a[i])

        # Make annotations
        annotations = DataFrame(index=scores.index)
        # Make IC(MoE)s
        annotations['IC(\u0394)'] = scores[['Score', '0.95 MoE']].apply(
            lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)
        # Make p-value
        annotations['p-value'] = scores['p-value'].apply('{:.2e}'.format)
        # Make FDRs
        annotations['FDR'] = scores['FDR'].apply('{:.2e}'.format)

        # Plot features title
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

        # Plot match panel
        plot_match_panel(target, target_int_to_o, features, max_std,
                         annotations, None, target_ax, features_ax,
                         target_type, data_type, None,
                         target_annotation_kwargs, plot_sample_names
                         and fi == len(multiple_features) - 1, None, dpi)

    if file_path:
        save_plot(file_path)
