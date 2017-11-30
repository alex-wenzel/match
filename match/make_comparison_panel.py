from numpy import array
from pandas import DataFrame

from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .nd_array.nd_array.apply_function_on_2_2d_arrays_slices import \
    apply_function_on_2_2d_arrays_slices
from .plot.plot.plot_clustermap import plot_clustermap
from .plot.plot.style import FIGURE_SIZE


def make_comparison_panel(array_2d_0,
                          array_2d_1,
                          function=compute_information_coefficient,
                          axis=0,
                          figure_size=FIGURE_SIZE,
                          annotate='auto',
                          title=None,
                          array_2d_0_name='',
                          array_2d_1_name='',
                          file_path_prefix=None):
    """
    Compare array_2d_0 and array_2d_1 slices and plot the result as clustermap.
    Arguments:
        array_2d_0 (array | DataFrame):
        array_2d_1 (array | DataFrame):
        function (callable):
        axis (int): 0 | 1
        figure_size (iterable):
        annotate (str | bool): whether to show values on the clustermap 'auto' |
            True | False
        title (str): plot title
        array_2d_0_name (str): array_2d_0 name
        array_2d_1_name (str): array_2d_1 name
        file_path_prefix (str): file_path_prefix.comparison_panel.tsv and
            file_path_prefix.comparison_panel.png will be saved
    Returns:
        array | DataFrame:
    """

    comparison = apply_function_on_2_2d_arrays_slices(
        array(array_2d_0), array(array_2d_1), function, axis=axis)

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
        comparison.to_csv(
            '{}.comparison_panel.tsv'.format(file_path_prefix), sep='\t')
        plot_file_path = '{}.comparison_panel.png'.format(file_path_prefix)
    else:
        plot_file_path = None

    plot_clustermap(
        comparison,
        figure_size=figure_size,
        annotate=annotate,
        title=title,
        xlabel=array_2d_1_name,
        ylabel=array_2d_0_name,
        file_path=plot_file_path)

    return comparison
