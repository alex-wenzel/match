from numpy import asarray
from pandas import DataFrame

from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .nd_array.nd_array.apply_function_on_2_2d_arrays_slices import \
    apply_function_on_2_2d_arrays_slices
from .plot.plot.plot_heat_map import plot_heat_map
from .support.support.path import (combine_path_prefix_and_suffix,
                                   establish_path)


def make_comparison_panel(
    _2d_array_or_df_0,
    _2d_array_or_df_1,
    match_function=compute_information_coefficient,
    axis=0,
    title="Comparison Panel",
    name_0=None,
    name_1=None,
    file_path_prefix=None,
    plotly_file_path_prefix=None,
):

    comparison = apply_function_on_2_2d_arrays_slices(
        asarray(_2d_array_or_df_0), asarray(_2d_array_or_df_1), match_function, axis
    )

    if isinstance(_2d_array_or_df_0, DataFrame) and isinstance(
        _2d_array_or_df_1, DataFrame
    ):

        if axis == 0:

            comparison = DataFrame(
                comparison,
                index=_2d_array_or_df_0.columns,
                columns=_2d_array_or_df_1.columns,
            )

        elif axis == 1:

            comparison = DataFrame(
                comparison,
                index=_2d_array_or_df_0.index,
                columns=_2d_array_or_df_1.index,
            )

    if file_path_prefix:

        establish_path(file_path_prefix, "file")

    if file_path_prefix is not None:

        comparison.to_csv("{}.tsv".format(file_path_prefix), sep="\t")

    suffix = ".html"

    plot_heat_map(
        comparison,
        cluster_axis="01",
        title=title,
        xaxis_title=name_1,
        yaxis_title=name_0,
        html_file_path=combine_path_prefix_and_suffix(file_path_prefix, suffix, False),
        plotly_file_path=combine_path_prefix_and_suffix(
            plotly_file_path_prefix, suffix, False
        ),
    )

    return comparison
