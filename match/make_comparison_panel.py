from numpy import array
from pandas import DataFrame

from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .nd_array.nd_array.apply_function_on_2_2d_arrays_slices import \
    apply_function_on_2_2d_arrays_slices
from .plot.plot.plot_heat_map import plot_heat_map
from .support.support.path import establish_path


def make_comparison_panel(array_2d_0,
                          array_2d_1,
                          match_function=compute_information_coefficient,
                          axis=0,
                          title='Comparison Panel',
                          array_2d_0_name='',
                          array_2d_1_name='',
                          file_path_prefix=None):

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
        html_file_path = '{}.comparison_panel.png'.format(file_path_prefix)

    else:
        html_file_path = None

    plot_heat_map(
        comparison,
        cluster=True,
        title=title,
        xaxis_title=array_2d_1_name,
        yaxis_title=array_2d_0_name,
        html_file_path=html_file_path)

    return comparison
