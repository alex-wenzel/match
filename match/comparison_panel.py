from .dataplay.dataplay.a2d import apply_2
from .information.information.information import information_coefficient
from .plot.plot.plot import plot_clustermap


def make_comparison_panel(a2d0,
                          a2d1,
                          function=information_coefficient,
                          axis=0,
                          is_distance=False,
                          annotate=True,
                          figure_size=(8, 8),
                          title=None,
                          a2d0_name='',
                          a2d1_name='',
                          file_path_prefix=None):
    """
    Compare a2d0 and a2d1 slices and plot the result as clustermap.
    :param a2d0: DataFrame | array;
    :param a2d1: DataFrame | array;
    :param function: callable; association or distance function
    :param axis: int; 0 | 1
    :param is_distance: bool; use distances: distances = 1 - associations or
        not
    :param annotate: bool; show values in the clustermap cells or not
    :param title: str; plot title
    :param a2d0_name: str; a2d0 name
    :param a2d1_name: str; a2d1 name
    :param file_path_prefix: str; file_path_prefix.comparison.txt and
        file_path_prefix.comparison.pdf will be saved
    :return: DataFrame | array; associations or distances
    """

    # Compute association or distance matrix, which is returned at the end
    comparison = apply_2(a2d1, a2d0, function, axis=axis)

    if is_distance:
        pass

    if file_path_prefix:  # Save
        comparison.to_csv(
            '{}.comparison.txt'.format(file_path_prefix), sep='\t')
        plot_file_path = '{}.comparison.pdf'.format(file_path_prefix)
    else:
        plot_file_path = None

    plot_clustermap(
        comparison,
        title=title,
        xlabel=a2d0_name,
        ylabel=a2d1_name,
        annotate=annotate,
        file_path=plot_file_path)

    return comparison
