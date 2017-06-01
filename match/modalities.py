def differential_gene_expression(phenotypes,
                                 gene_expression,
                                 output_filename,
                                 max_number_of_genes_to_show=20,
                                 number_of_permutations=10,
                                 title=None,
                                 random_seed=RANDOM_SEED):
    """
    Sort genes according to their association with a binary phenotype or class vector.
    :param phenotypes: Series; input binary phenotype/class distinction
    :param gene_expression: Dataframe; data matrix with input gene expression profiles
    :param output_filename: str; output files will have this name plus extensions .txt and .pdf
    :param max_number_of_genes_to_show: int; maximum number of genes to show in the heatmap
    :param number_of_permutations: int; number of random permutations to estimate statistical significance (p-values and FDRs)
    :param title: str;
    :param random_seed: int | array; random number generator seed (can be set to a user supplied integer for reproducibility)
    :return: Dataframe; table of genes ranked by Information Coeff vs. phenotype
    """
    gene_scores = make_match_panel(
        target=phenotypes,
        features=gene_expression,
        n_jobs=1,
        max_n_features=max_number_of_genes_to_show,
        n_permutations=number_of_permutations,
        target_type='binary',
        title=title,
        file_path_prefix=output_filename,
        random_seed=random_seed)
    return gene_scores
