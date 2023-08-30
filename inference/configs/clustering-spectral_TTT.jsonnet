local tuned = import 'clustering-algorithm/hyperopt-tuned-clustering-algorithm.jsonnet';
local label_propagated = import 'clustering-algorithm/label-propagated-clustering-algorithm.jsonnet';
local precomputed = import 'clustering-algorithm/precomputed-distances-wrapper.jsonnet';

local spectral = import 'clustering-algorithm/spectral.jsonnet';

local tuned_spectral = tuned(
    clustering_algorithm = spectral(),
    parameter_search_space = {
        n_clusters: ['quniform', 5, 50, 1]
        // n_clusters: ['quniform', 17, 50, 1]
        //n_clusters: ['quniform', 34, 50, 1]
        //n_clusters: ['quniform', 49, 50, 1] 
        //n_clusters: ['qnormal', 27.5, 7.5, 1]
    },
    // spectral results may differ slightly by seed, so take average score over 3 trials
    trials_per_eval = 3,
    // number of trials without improvement before early sotpping
    patience = 25,
);

{
    spectral: {name: 'spectral_TTT', model: tuned_spectral},
}