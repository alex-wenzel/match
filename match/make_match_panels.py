from os.path import isfile

from pandas import read_table

from .make_match_panel import make_match_panel
from .plot.plot.make_html_and_plotly_file_paths import \
    make_html_and_plotly_file_paths


def make_match_panels(
        multiple_target,
        multiple_drop_negative_target,
        multiple_target_ascending,
        multiple_target_type,
        feature_dicts,
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

        print(target.name)

        if drop_negative_target:

            target = target[target != -1]

        for feature_name, feature_dict in feature_dicts.items():

            suffix = '{}/{}'.format(
                target.name,
                feature_name,
            )

            print('{} ...'.format(suffix))

            file_path_prefix, plotly_file_path_prefix = make_html_and_plotly_file_paths(
                suffix,
                directory_path,
                plotly_directory_path,
                prefix_is_directory=True,
            )

            scores_file_path = '{}.tsv'.format(file_path_prefix)

            if not overwrite and isfile(scores_file_path):

                print('Reading scores from {} ...'.format(scores_file_path))

                scores = read_table(
                    scores_file_path,
                    index_col=0,
                )

            else:

                scores = None

            make_match_panel(
                target,
                feature_dict['df'],
                target_ascending=target_ascending,
                scores=scores,
                scores_ascending=feature_dict['emphasis'] == 'low',
                target_type=target_type,
                features_type=feature_dict['data_type'],
                title=suffix.replace(
                    '/',
                    ' vs ',
                ),
                file_path_prefix=file_path_prefix,
                plotly_file_path_prefix=plotly_file_path_prefix,
                **kwargs,
            )
