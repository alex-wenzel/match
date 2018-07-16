from os.path import isfile

from pandas import read_table

from ._check_features_index import _check_features_index
from .make_match_panel import make_match_panel
from .plot.plot.make_html_and_plotly_file_paths import \
    make_html_and_plotly_file_paths


def make_match_panels(
        targets,
        feature_dicts,
        n_job=1,
        extreme_feature_threshold=16,
        n_sampling=0,
        n_permutation=0,
        target_type='continuous',
        plot_target_std_max=3,
        plot_features_std_max=3,
        directory_path=None,
        plotly_directory_path=None,
        overwrite=False,
):

    for target_index, target in targets.iterrows():

        for feature_name, feature_dict in feature_dicts.items():

            features = feature_dict['df']

            _check_features_index(features)

            file_path_prefix, plotly_file_path_prefix = make_html_and_plotly_file_paths(
                '{}/{}'.format(
                    target.name,
                    feature_name,
                ),
                directory_path,
                plotly_directory_path,
                prefix_is_directory=True,
            )

            print('{} ...'.format(file_path_prefix))

            scores_file_path = '{}.tsv'.format(file_path_prefix)

            if overwrite or not isfile(scores_file_path):

                scores = None

            else:

                print('Reading scores from {} ...'.format(scores_file_path))

                scores = read_table(
                    scores_file_path,
                    index_col=0,
                )

            if feature_dict['emphasis'] == 'high':

                scores_ascending = False

            elif feature_dict['emphasis'] == 'low':

                scores_ascending = True

            make_match_panel(
                target,
                features,
                scores=scores,
                n_job=n_job,
                scores_ascending=scores_ascending,
                extreme_feature_threshold=extreme_feature_threshold,
                n_sampling=n_sampling,
                n_permutation=n_permutation,
                target_type=target_type,
                features_type=feature_dict['data_type'],
                plot_target_std_max=plot_target_std_max,
                plot_features_std_max=plot_features_std_max,
                title=feature_name,
                file_path_prefix=file_path_prefix,
                plotly_file_path_prefix=plotly_file_path_prefix,
            )
