local spectral(use_reference_n_clusters = false) = {
    type: 'sklearn_clustering_algorithm',
    clustering_algorithm_name: 'spectral',
    clustering_algorithm_params: {
        n_init: 10,
        affinity:'nearest_neighbors',
        // assign_labels: 'discretize',
        
    },
};

spectral