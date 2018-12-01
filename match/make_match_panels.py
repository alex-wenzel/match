from os.path import isfile

from pandas import read_table

from .make_match_panel import make_match_panel
from .support.support.path import combine_path_prefix_and_suffix


def make_match_panels(
    target_x_sample,
    feature_dicts,
    drop_negative_target=False,
    directory_path=None,
    plotly_directory_path=None,
    read_score_moe_p_value_fdr=False,
    **kwargs,
):

    for target_name, target_values in target_x_sample.iterrows():

        if drop_negative_target:

            target_values = target_values[target_values != -1]

        for feature_group, feature_dict in feature_dicts.items():

            suffix = "{}/{}".format(target_name, feature_group)

            print("Making match panel for {} ...".format(suffix))

            file_path_prefix = combine_path_prefix_and_suffix(
                directory_path, suffix, True
            )

            scores_file_path = "{}.tsv".format(file_path_prefix)

            if read_score_moe_p_value_fdr and isfile(scores_file_path):

                print(
                    "Reading score_moe_p_value_fdr from {} ...".format(scores_file_path)
                )

                score_moe_p_value_fdr = read_table(scores_file_path, index_col=0)

            else:

                score_moe_p_value_fdr = None

            make_match_panel(
                target_values,
                feature_dict["df"],
                score_moe_p_value_fdr=score_moe_p_value_fdr,
                features_type=feature_dict["data_type"],
                title=suffix.replace("/", "<br>"),
                file_path_prefix=file_path_prefix,
                plotly_file_path_prefix=combine_path_prefix_and_suffix(
                    plotly_directory_path, suffix, True
                ),
                **kwargs,
            )
