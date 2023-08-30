local tuned = import 'clustering-algorithm/hyperopt-tuned-clustering-algorithm.jsonnet';
local label_propagated = import 'clustering-algorithm/label-propagated-clustering-algorithm.jsonnet';
local precomputed = import 'clustering-algorithm/precomputed-distances-wrapper.jsonnet';

local agglomerative = import 'clustering-algorithm/agglomerative.jsonnet';

local tuned_agglomerative = tuned(
    clustering_algorithm = agglomerative(),
    parameter_search_space = {
        n_clusters: ['quniform', 5, 50, 1]
    },
    // agglomerative results may differ slightly by seed, so take average score over 3 trials
    trials_per_eval = 3,
    // number of trials without improvement before early sotpping
    patience = 25,
);

{
    agglomerative: {name: 'agglomerative', model: tuned_agglomerative},
}