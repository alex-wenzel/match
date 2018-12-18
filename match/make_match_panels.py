from os.path import isfile

from pandas import read_table

from .make_match_panel import make_match_panel
from .support.support.path import combine_path_prefix_and_suffix
from .support.support.str_ import make_file_name_from_str


def make_match_panels(
    target_x_sample,
    data_dicts,
    drop_negative_target=False,
    directory_path=None,
    plotly_directory_path=None,
    read_score_moe_p_value_fdr=False,
    **kwargs,
):

    for target_name, target_values in target_x_sample.iterrows():

        if drop_negative_target:

            target_values = target_values[target_values != -1]

        for data_name, data_dict in data_dicts.items():

            suffix = "{}/{}".format(target_name, make_file_name_from_str(data_name))

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
                data_dict["df"],
                score_moe_p_value_fdr=score_moe_p_value_fdr,
                features_type=data_dict["data_type"],
                score_ascending=data_dict.get("emphasis", "high") == "low",
                title=suffix.replace("/", "<br>"),
                file_path_prefix=file_path_prefix,
                plotly_file_path_prefix=combine_path_prefix_and_suffix(
                    plotly_directory_path, suffix, True
                ),
                **kwargs,
            )
