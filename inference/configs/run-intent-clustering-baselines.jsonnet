local baseline_intent_clustering_model_fn = import 'intent-clustering/baseline-intent-clustering-model.jsonnet';
local clustering_baselines = import 'clustering-baselines.jsonnet';

local clustering_kmeans = import 'clustering-kmeans.jsonnet';
local clustering_spectral = import 'clustering-spectral.jsonnet';
local clustering_gmm = import 'clustering-gmm.jsonnet';
local clustering_agglomerative = import 'clustering-agglomerative.jsonnet';

local clustering_spectral_FTF = import 'clustering-spectral_FTF.jsonnet';
local clustering_spectral_FTT = import 'clustering-spectral_FTT.jsonnet';
local clustering_spectral_TTF = import 'clustering-spectral_TTF.jsonnet';
local clustering_spectral_TTT = import 'clustering-spectral_TTT.jsonnet';
local clustering_spectral_TFT = import 'clustering-spectral_TFT.jsonnet';
local clustering_spectral_FFT = import 'clustering-spectral_FFT.jsonnet';

local clustering_kmeans_FTF = import 'clustering-kmeans_FTF.jsonnet';
local clustering_kmeans_FTT = import 'clustering-kmeans_FTT.jsonnet';
local clustering_kmeans_TTF = import 'clustering-kmeans_TTF.jsonnet';
local clustering_kmeans_TTT = import 'clustering-kmeans_TTT.jsonnet';
local clustering_kmeans_TFT = import 'clustering-kmeans_TFT.jsonnet';
local clustering_kmeans_FFT = import 'clustering-kmeans_FFT.jsonnet';


local all_mpnet = {key: 'all-mpnet-base-v2', model: import 'embedding-model/all-mpnet-base-v2.jsonnet'};

local intent_clustering_experiment(run_id, intent_clustering_model) = {
    type: 'intent_clustering_experiment',
    run_id: run_id,
    dialogue_reader: 'default_dialogue_reader',
    dialogues_path: 'dialogues.jsonl',
    intent_clustering_model: intent_clustering_model
};

local ic_exp(baseline, embedding_model) = intent_clustering_experiment(
    baseline.name + '_' + embedding_model.key,
    baseline_intent_clustering_model_fn(
        clustering_algorithm = baseline.model,
        embedding_model = embedding_model.model,
    )
);

{
    type: 'meta_experiment',
    run_id: 'T1  ' + all_mpnet.model.prefix + '_quniform_5-50_discretize',
    datasets: [
        'development',
        'test-banking',
        'test-finance',
    ],

    experiments:  [
        // ic_exp(clustering_spectral.spectral, all_mpnet),

        // ic_exp(clustering_spectral_FTF.spectral, all_mpnet),
        // ic_exp(clustering_spectral_FTT.spectral, all_mpnet),
        // ic_exp(clustering_spectral_TTF.spectral, all_mpnet),
        // ic_exp(clustering_spectral_TTT.spectral, all_mpnet),
        // ic_exp(clustering_spectral_TFT.spectral, all_mpnet),
        // ic_exp(clustering_spectral_FFT.spectral, all_mpnet),

        // ic_exp(clustering_kmeans.kmeans, all_mpnet),

        // ic_exp(clustering_kmeans_FTF.kmeans, all_mpnet),
        // ic_exp(clustering_kmeans_FTT.kmeans, all_mpnet),
        // ic_exp(clustering_kmeans_TTF.kmeans, all_mpnet),
        // ic_exp(clustering_kmeans_TTT.kmeans, all_mpnet),
        // ic_exp(clustering_kmeans_TFT.kmeans, all_mpnet),
        // ic_exp(clustering_kmeans_FFT.kmeans, all_mpnet),

    ],
}