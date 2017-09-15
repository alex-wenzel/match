from numpy import array
from pandas import DataFrame

from .array_nd.array_nd.array_2d import apply_2
from .information.information.information import \
    compute_information_coefficient
from .plot.plot.plot import plot_clustermap
from .plot.plot.style import FIGURE_SIZE


def make_comparison_panel(a2d0,
                          a2d1,
                          function=compute_information_coefficient,
                          axis=0,
                          figure_size=FIGURE_SIZE,
                          title=None,
                          a2d0_name='',
                          a2d1_name='',
                          file_path_prefix=None):
    """
    Compare a2d0 and a2d1 slices and plot the result as clustermap.
    Arguments:
        a2d0 (array | DataFrame):
        a2d1 (array | DataFrame):
        function (callable):
        axis (int): 0 | 1
        annotate (bool): Whether to show values on the clustermap
        title (str): Plot title
        a2d0_name (str): a2d0 name
        a2d1_name (str): a2d1 name
        file_path_prefix (str): file_path_prefix.comparison_panel.txt and
            file_path_prefix.comparison_panel.png will be saved
    Returns:
        array | DataFrame:
    """

    comparison = apply_2(array(a2d1), array(a2d0), function, axis=axis)

    if isinstance(a2d0, DataFrame):

        if axis == 0:
            comparison = DataFrame(
                comparison, index=a2d1.columns, columns=a2d0.columns)

        elif axis == 1:
            comparison = DataFrame(
                comparison, index=a2d1.index, columns=a2d0.index)

    if file_path_prefix:
        comparison.to_csv(
            '{}.comparison_panel.txt'.format(file_path_prefix), sep='\t')
        plot_file_path = '{}.comparison_panel.png'.format(file_path_prefix)

    else:
        plot_file_path = None

    plot_clustermap(
        comparison,
        figure_size=figure_size,
        title=title,
        xlabel=a2d0_name,
        ylabel=a2d1_name,
        file_path=plot_file_path)

    return comparison
