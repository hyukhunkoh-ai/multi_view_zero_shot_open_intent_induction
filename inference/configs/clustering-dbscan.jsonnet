local tuned = import 'clustering-algorithm/hyperopt-tuned-clustering-algorithm.jsonnet';
local label_propagated = import 'clustering-algorithm/label-propagated-clustering-algorithm.jsonnet';
local precomputed = import 'clustering-algorithm/precomputed-distances-wrapper.jsonnet';

local dbscan = import 'clustering-algorithm/dbscan.jsonnet';

local tuned_dbscan = tuned(
    clustering_algorithm = dbscan(),
    parameter_search_space = {
        eps: ['quniform', 5, 50, 1]
    },
    // dbscan results may differ slightly by seed, so take average score over 3 trials
    trials_per_eval = 3,
    // number of trials without improvement before early sotpping
    patience = 25,
);

{
    dbscan: {name: 'dbscan', model: tuned_dbscan},
}