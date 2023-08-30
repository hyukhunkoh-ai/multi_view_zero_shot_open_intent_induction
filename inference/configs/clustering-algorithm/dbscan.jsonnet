local dbscan(use_reference_n_clusters = false) = {
    type: 'sklearn_clustering_algorithm',
    clustering_algorithm_name: 'dbscan',
    clustering_algorithm_params: {
        min_samples: 20,
    },
};

dbscan