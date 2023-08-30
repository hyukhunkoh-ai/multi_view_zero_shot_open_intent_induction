local agglomerative(use_reference_n_clusters = false) = {
    type: 'sklearn_clustering_algorithm',
    clustering_algorithm_name: 'agglomerative',
    clustering_algorithm_params: {
        affinity: 'cosine',
        linkage: 'complete'
    },
};

agglomerative