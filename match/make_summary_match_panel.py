from matplotlib.colorbar import ColorbarBase, make_axes
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from pandas import DataFrame, read_table

from .plot.plot.save_plot import save_plot
from .plot.plot.style import FIGURE_SIZE, FONT_LARGER, FONT_LARGEST
from .plot_match import plot_match
from .prepare_data_for_plotting import prepare_data_for_plotting
from .preprocess_target_and_features import preprocess_target_and_features


def make_summary_match_panel(target,
                             multiple_features,
                             indexs=(),
                             repeat_plotting_target=True,
                             target_ascending=False,
                             max_n_unique_objects_for_drop_slices=1,
                             result_in_ascending_order=False,
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
            Emphasis (str): 'High' | 'Low',
            Feature type (str): 'continuous' | 'categorical' | 'binary',
            Match (str | DataFrame): Saved file path or returned DataFrame from
                make_match_panel
            Index (iterable): Features to plot,
            Index alias (iterable): Name shown for the features to plot,
        ]
        indexs (iterable | str): iterable (n_samples_to_plot) | () (for plotting
            columns shared between target and each feature) | 'only_shared' (for
            plotting columns shared between target and all features)
        repeat_plotting_target (bool): Whether to repeat plotting target
        target_ascending (bool): True if target increase from left to right,
            and False right to left
        max_n_unique_objects_for_drop_slices (int):
        result_in_ascending_order (bool): True if result increase from top to
            bottom, and False bottom to top
        target_type (str): 'continuous' | 'categorical' | 'binary'
        features_type (str): 'continuous' | 'categorical' | 'binary'
        title (str): Plot title
        plot_sample_names (bool): Whether to plot column names
        file_path (str):
    Returns:
        None
    """

    # TODO: Compute inplace

    # Set up figure
    fig = figure(figsize=FIGURE_SIZE)

    # Compute the number of row-grids for setting up a figure
    n = 0
    for name, features, emphasis, features_type, scores, index, alias in multiple_features:
        n += len(index) + 3

    # Add a row for color bar
    n += 1

    # Set up axis grids
    gridspec = GridSpec(n, 1)

    # Annotate target with features
    r_i = 0
    fig.suptitle(title, horizontalalignment='center', **FONT_LARGEST)

    if indexs == 'only_shared':
        for name, features, emphasis, features_type, scores, index, alias in multiple_features:
            if indexs is 'only_shared':
                indexs = features.columns
            else:
                indexs &= features.columns
    print('Indexs: {}'.format(indexs))

    for fi, (name, features, emphasis, features_type, scores, index,
             alias) in enumerate(multiple_features):

        target, features = preprocess_target_and_features(
            target,
            features,
            target_ascending,
            max_n_unique_objects_for_drop_slices,
            indexs=indexs)

        # Prepare target for plotting
        target, target_min, target_max, target_cmap = prepare_data_for_plotting(
            target, target_type)

        # Prepare features for plotting
        features, features_min, features_max, features_cmap = prepare_data_for_plotting(
            features, features_type)

        # Read corresponding match score file
        if isinstance(scores, str):
            scores = read_table(scores, index_col=0)

        # Keep only selected features
        scores = scores.loc[index]

        # Sort by match score
        scores.sort_values(
            'Score', ascending=result_in_ascending_order, inplace=True)

        # Apply the sorted index to featuers
        features = features.loc[scores.index]

        i_to_a = {i: a for i, a in zip(index, alias)}
        features.index = features.index.map(lambda i: i_to_a[i])

        print('Making annotations ...')
        annotations = DataFrame(index=scores.index)
        # Add IC(confidence interval), p-value, and FDR
        annotations['IC(\u0394)'] = scores[['Score', '0.95 CI']].apply(
            lambda s: '{0:.3f}({1:.3f})'.format(*s), axis=1)
        annotations['p-value'] = scores['p-value'].apply('{:.2e}'.format)
        annotations['FDR'] = scores['FDR'].apply('{:.2e}'.format)

        #
        # Set up axes
        #
        r_i += 1
        title_ax = subplot(gridspec[r_i:r_i + 1, 0])
        title_ax.axis('off')

        # Plot title
        title_ax.text(
            title_ax.axis()[1] / 2,
            -title_ax.axis()[2] / 2,
            '{} (n={})'.format(name, target.size),
            horizontalalignment='center',
            **FONT_LARGER)

        if fi == 0 or repeat_plotting_target:
            r_i += 1
            target_ax = subplot(gridspec[r_i:r_i + 1, 0])
        else:
            target_ax = False
        r_i += 1
        features_ax = subplot(gridspec[r_i:r_i + features.shape[0], 0])

        r_i += features.shape[0]

        # Plot match
        plot_match(
            target,
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

        # Plot colorbar
        if r_i == n - 1:
            colorbar_ax = subplot(gridspec[r_i:r_i + 1, 0])
            colorbar_ax.axis('off')
            cax, kw = make_axes(
                colorbar_ax,
                location='bottom',
                pad=0.026,
                fraction=0.26,
                shrink=2.6,
                aspect=26,
                cmap=target_cmap,
                norm=Normalize(-3, 3),
                ticks=range(-3, 4, 1))
            ColorbarBase(cax, **kw)

    if file_path:
        save_plot(file_path)
