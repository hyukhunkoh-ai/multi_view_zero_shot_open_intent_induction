local tuned = import 'clustering-algorithm/hyperopt-tuned-clustering-algorithm.jsonnet';
local label_propagated = import 'clustering-algorithm/label-propagated-clustering-algorithm.jsonnet';
local precomputed = import 'clustering-algorithm/precomputed-distances-wrapper.jsonnet';

local gmm = import 'clustering-algorithm/gmm.jsonnet';

local tuned_gmm = tuned(
    clustering_algorithm = gmm(),
    parameter_search_space = {
        n_components: ['quniform', 5, 50, 1]
    },
    // gmm results may differ slightly by seed, so take average score over 3 trials
    trials_per_eval = 3,
    // number of trials without improvement before early sotpping
    patience = 25,
);

{
    gmm: {name: 'gmm', model: tuned_gmm},
}