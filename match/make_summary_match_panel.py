from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from pandas import DataFrame

from .plot.plot.save_plot import save_plot
from .plot.plot.style import FIGURE_SIZE, FONT_LARGER, FONT_LARGEST
from .plot_match import plot_match
from .prepare_data_for_plotting import prepare_data_for_plotting
from .preprocess_target_and_features import preprocess_target_and_features


def make_summary_match_panel(target,
                             multiple_features,
                             indexs=(),
                             target_ascending=False,
                             max_n_unique_objects_for_drop_slices=1,
                             target_type='continuous',
                             features_type='continuous',
                             title='Summary Match Panel',
                             plot_sample_names=False,
                             file_path=None):
    """
    Make summary match panel.
    Arguments:
        target (iterable): (n_samples)
        multiple_features (iterable): [
            Feature name (str):,
            Features (DataFrame): (n_features, n_samples),
            Increasing (bool): True (scores increase from top to bottom) | False,
            Feature type (str): 'continuous' | 'categorical' | 'binary',
            Scores (DataFrame): Scores DataFrame returned from make_match_panel,
            Index (iterable): Features to plot,
            Index alias (iterable): Name shown for the features to plot,
        ]
        indexs (iterable | str): iterable (n_samples_to_plot) | () (for plotting
            columns shared between target and each feature) |
            'shared_by_target_and_all_features' (for plotting columns shared
            among target and all features)
        target_ascending (bool): True if target increase from left to right,
            and False right to left
        max_n_unique_objects_for_drop_slices (int):
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        title (str): Plot title
        plot_sample_names (bool): Whether to plot column names
        file_path (str):
    Returns:
        None
    """

    target_o_to_int = {}
    target_int_to_o = {}

    if target.dtype == 'O':

        for i, o in enumerate(target.unique()):
            target_o_to_int[o] = i
            target_int_to_o[i] = o

        # Make target numerical
        target = target.map(target_o_to_int)

    # Set up figure
    fig = figure(figsize=FIGURE_SIZE)

    # Compute the number of row-grids for setting up a figure
    n = 0
    for f in multiple_features:
        n += len(f[5]) + 3

    # Set up axis grids
    gridspec = GridSpec(n, 1)

    # Annotate target with features
    r_i = 0
    fig.suptitle(title, horizontalalignment='center', **FONT_LARGEST)

    if indexs == 'shared_by_target_and_all_features':

        indexs = None
        for f in multiple_features:
            if indexs is None:
                indexs = f[1].columns
            else:
                indexs &= f[1].columns

    print('Indexs: {}'.format(indexs))

    for fi, (name, features, increasing, features_type, scores, index,
             alias) in enumerate(multiple_features):

        target, features = preprocess_target_and_features(
            target, features, indexs, target_ascending,
            max_n_unique_objects_for_drop_slices)

        # Prepare target for plotting
        target, target_min, target_max, target_cmap = prepare_data_for_plotting(
            target, target_type)

        # Prepare features for plotting
        features, features_min, features_max, features_cmap = prepare_data_for_plotting(
            features, features_type)

        # Keep only selected features and sort by match score
        scores = scores.loc[index].sort_values('Score', ascending=increasing)

        # Apply the sorted index to featuers
        features = features.loc[scores.index]

        i_to_a = {i: a for i, a in zip(index, alias)}
        features.index = features.index.map(lambda i: i_to_a[i])

        print('Making annotations ...')
        annotations = DataFrame(index=scores.index)

        # Add IC(confidence interval)
        annotations['IC(\u0394)'] = scores[['Score', '0.95 CI']].apply(
            lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)

        # Add p-value
        annotations['p-value'] = scores['p-value'].apply('{:.2e}'.format)

        # Add FDR
        annotations['FDR'] = scores['FDR'].apply('{:.2e}'.format)

        # Plot title
        r_i += 1
        title_ax = subplot(gridspec[r_i:r_i + 1, 0])
        title_ax.axis('off')

        title_ax.text(
            title_ax.axis()[1] / 2,
            0,
            '{} (n={})'.format(name, target.size),
            horizontalalignment='center',
            **FONT_LARGER)

        r_i += 1
        target_ax = subplot(gridspec[r_i:r_i + 1, 0])

        r_i += 1
        features_ax = subplot(gridspec[r_i:r_i + features.shape[0], 0])

        r_i += features.shape[0]

        # Plot match
        plot_match(
            target,
            target_int_to_o,
            features,
            annotations,
            None,
            target_type,
            features_type,
            None,
            plot_sample_names and fi == len(multiple_features) - 1,
            None,
            target_ax=target_ax,
            features_ax=features_ax)

    if file_path:
        save_plot(file_path)
