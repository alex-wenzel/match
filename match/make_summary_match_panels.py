from . import make_summary_match_panel
from .plot.plot.make_html_and_plotly_file_paths import \
    make_html_and_plotly_file_paths


def make_summary_match_panels(
        multiple_target,
        multiple_drop_negative_target,
        multiple_target_ascending,
        multiple_target_type,
        title_feature_dicts,
        directory_path=None,
        plotly_directory_path=None,
        overwrite=True,
        **kwargs,
):

    for target, drop_negative_target, target_ascending, target_type in zip(
            multiple_target,
            multiple_drop_negative_target,
            multiple_target_ascending,
            multiple_target_type,
    ):

        if drop_negative_target:

            target = target[target != -1]

        for title, feature_dicts in title_feature_dicts.items():

            suffix = '{}/{}'.format(
                target.name,
                title,
            )

            print('Making summary match panel for {} ...'.format(suffix))

            html_file_path, plotly_file_path = make_html_and_plotly_file_paths(
                '{}.html'.format(suffix),
                directory_path,
                plotly_directory_path,
                prefix_is_directory=True,
            )

            make_summary_match_panel(
                target,
                feature_dicts,
                target_ascending=target_ascending,
                target_type=target_type,
                title=suffix.replace(
                    '/',
                    ' & ',
                ),
                html_file_path=html_file_path,
                plotly_file_path=plotly_file_path,
                **kwargs,
            )
