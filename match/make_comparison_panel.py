from numpy import array
from pandas import DataFrame

from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .nd_array.nd_array.apply_function_on_2_2d_arrays_slices import \
    apply_function_on_2_2d_arrays_slices
from .plot.plot.plot_clustermap import plot_clustermap
from .plot.plot.style import CMAP_CONTINUOUS_BWR, FIGURE_SIZE
from .support.support.path import establish_path


def make_comparison_panel(array_2d_0,
                          array_2d_1,
                          match_function=compute_information_coefficient,
                          axis=0,
                          figure_size=FIGURE_SIZE,
                          title='',
                          array_2d_0_name='',
                          array_2d_1_name='',
                          file_path_prefix=None):
    """
    Compare array_2d_0 and array_2d_1 slices and plot the result as clustermap.
    Arguments:
        array_2d_0 (ndarray | DataFrame):
        array_2d_1 (ndarray | DataFrame):
        match_function (callable):
        axis (int): 0 | 1
        figure_size (iterable):
        title (str):
        array_2d_0_name (str):
        array_2d_1_name (str):
        file_path_prefix (str):
    Returns:
        ndarray | DataFrame:
    """

    comparison = apply_function_on_2_2d_arrays_slices(
        array(array_2d_0), array(array_2d_1), match_function, axis=axis)

    if isinstance(array_2d_0, DataFrame):

        if axis == 0:
            comparison = DataFrame(
                comparison,
                index=array_2d_0.columns,
                columns=array_2d_1.columns)

        elif axis == 1:
            comparison = DataFrame(
                comparison, index=array_2d_0.index, columns=array_2d_1.index)

    if file_path_prefix:
        establish_path(file_path_prefix, 'file')
        comparison.to_csv(
            '{}.comparison_panel.tsv'.format(file_path_prefix), sep='\t')
        plot_file_path = '{}.comparison_panel.png'.format(file_path_prefix)
    else:
        plot_file_path = None

    plot_clustermap(
        comparison,
        clustermap_kwargs={
            'figsize': figure_size,
            'cmap': CMAP_CONTINUOUS_BWR,
        },
        title=title,
        decorate_ax_kwargs={
            'xlabel': array_2d_1_name,
            'ylabel': array_2d_0_name,
        },
        file_path=plot_file_path)

    return comparison
