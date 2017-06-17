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


def expand_gene_set(gene_sets,
                    gene_sets_by_n_columns,
                    gene_by_sample_matrix,
                    gene_set_by_sample_matrix,
                    target,
                    target_phenotypes,
                    results_directory_path,
                    dropna='all',
                    target_ascending=False,
                    features_ascending=False,
                    n_jobs=1,
                    n_features=10,
                    n_samplings=30,
                    n_permutations=30,
                    target_name=None,
                    target_type='continuous',
                    features_type='continuous',
                    plot_colname=False):

    """
    Compute association of genes in given gene set.
    :param gene_sets: list of strs; ['gene_set_1', 'gene_set_2']
    :param gene_sets_by_n_columns: Dataframe; (gene_sets, n_columns); each row contains gene name strs of genes in gene set
    :param gene_by_sample_matrix: Dataframe; (genes, n_samples)
    :param gene_set_by_sample_matrix: Dataframe; (gene_sets, n_samples)
    :param target: Series; (n_samples); must have name and index matching features's column names
    :param target_phenotypes: list of strs; ['Phenotype 1', 'Phenotype 2']
    :param results_directory_path: str; path to directory where results will be stored
    :param dropna: str; 'any' or 'all'
    :param target_ascending: bool;
    :param features_ascending: bool; True if features scores increase from top to bottom, and False otherwise
    :param n_jobs: int; number of jobs to parallelize
    :param n_features: int or float; number threshold if >= 1, and percentile threshold if < 1
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :param target_name: str;
    :param target_type: str; {'continuous', 'categorical', 'binary'}
    :param features_type: str; {'continuous', 'categorical', 'binary'}
    :param plot_colname: bool; plot column names below the plot or not
    :return: .pdf and .txt ; (n_features, 8 ('score', '<confidence> moe',
                                        'p-value (forward)', 'p-value (reverse)', 'p-value',
                                        'fdr (forward)', 'fdr (reverse)', 'fdr'))
    """

    for gene_set in gene_sets:
        # Get gene names
        genes = []
        for n in range(0, 2940):
            gene = gene_sets_by_n_columns.ix[gene_set][n]
            if gene is not None:
                if str(gene) != 'nan':
                    genes.append(gene)
        print(genes)

        # Build gene by sample matrix
        gene_sample_list = []
        gs = gene_set_by_sample_matrix.ix[gene_set, :slightly_smiling_face:
        gene_sample_list.append(list(gene_set_by_sample_matrix.ix[gene_set, :]))
        for gene in genes:
            if gene in gene_by_sample_matrix.index:
                expression = gene_by_sample_matrix.ix[gene]
                gene_sample_list.append(list(expression))
            else:
                gene_to_remove = gene
        chosen_gene_sample = pd.DataFrame(gene_sample_list)
        genes.remove(gene_to_remove)
        chosen_gene_sample.index = list(chosen_gene_sample.index)
        genes.insert(0, gene_set)
        chosen_gene_sample.index = genes
        chosen_gene_sample.columns = list(chosen_gene_sample.columns)
        chosen_gene_sample.columns = gene_by_sample_matrix.columns

        # Perform association analysis on gene by sample matrix and show heatmap
        phenotypes = []
        for p in target_phenotypes:
            p = p.lower()
            p = p.replace(' ', '_')
            phenotypes.append(p)
        phenotypes = tuple(phenotypes)
        phenotypes = '_'.join(phenotypes)

        filename_pieces = (phenotypes, gene_set, 'gene_panel')
        filepath_prefix = '_'.join(filename_pieces)
        filepath_items = (results_directory_path, filepath_prefix)
        filepath_prefix = ''.join(filepath_items)

        title_phenotypes = ' vs '.join(target_phenotypes)
        title_pieces = (title_phenotypes, gene_set, 'Genes')
        title = ' '.join(title_pieces)

        ccal.association.make_association_panel(target,
                               chosen_gene_sample,
                               dropna=dropna,
                               target_ascending=target_ascending,
                               features_ascending=features_ascending,
                               n_jobs=n_jobs,
                               n_features=n_features,
                               n_samplings=n_samplings,
                               n_permutations=n_permutations,
                               target_name=target_name,
                               target_type=target_type,
                               features_type=features_type,
                               title=title,
                               plot_colname=plot_colname,
                               filepath_prefix=filepath_prefix)
