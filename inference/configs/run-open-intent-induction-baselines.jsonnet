local baseline_intent_clustering_model_fn = import 'intent-clustering/baseline-intent-clustering-model.jsonnet';
local baseline_open_intent_induction_model_fn = import 'open-intent-induction/baseline-open-intent-induction-model.jsonnet';
local logistic_regression_classifier_evaluator = import 'classifier-evaluator/logistic-regression-classifier-evaluator.jsonnet';

local logistic_regression_classifier_evaluator_FTF = import 'classifier-evaluator/logistic-regression-classifier-evaluator_FTF.jsonnet';
local logistic_regression_classifier_evaluator_FTT = import 'classifier-evaluator/logistic-regression-classifier-evaluator_FTT.jsonnet';
local logistic_regression_classifier_evaluator_TTF = import 'classifier-evaluator/logistic-regression-classifier-evaluator_TTF.jsonnet';
local logistic_regression_classifier_evaluator_TTT = import 'classifier-evaluator/logistic-regression-classifier-evaluator_TTT.jsonnet';
local logistic_regression_classifier_evaluator_TFT = import 'classifier-evaluator/logistic-regression-classifier-evaluator_TFT.jsonnet';
local logistic_regression_classifier_evaluator_FFT = import 'classifier-evaluator/logistic-regression-classifier-evaluator_FFT.jsonnet';

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

local all_mpnet_FTF = {key: 'all-mpnet-base-v2_FTF', model: import 'embedding-model/all-mpnet-base-v2_FTF.jsonnet'};
local all_mpnet_FTT = {key: 'all-mpnet-base-v2_FTT', model: import 'embedding-model/all-mpnet-base-v2_FTT.jsonnet'};
local all_mpnet_TTF = {key: 'all-mpnet-base-v2_TTF', model: import 'embedding-model/all-mpnet-base-v2_TTF.jsonnet'};
local all_mpnet_TTT = {key: 'all-mpnet-base-v2_TTT', model: import 'embedding-model/all-mpnet-base-v2_TTT.jsonnet'};
local all_mpnet_TFT = {key: 'all-mpnet-base-v2_TFT', model: import 'embedding-model/all-mpnet-base-v2_TFT.jsonnet'};
local all_mpnet_FFT = {key: 'all-mpnet-base-v2_FFT', model: import 'embedding-model/all-mpnet-base-v2_FFT.jsonnet'};

local baseline_open_intent_induction_experiment(
    run_id,
    intent_clustering_model,
    embedding_model = all_mpnet,
    classifier_evaluator = logistic_regression_classifier_evaluator(),
) = {
    type: 'open_intent_induction_experiment',
    run_id: run_id,
    dialogue_reader: 'default_dialogue_reader',
    dialogues_path: 'dialogues.jsonl',
    test_utterances_path: 'test-utterances.jsonl',
    open_intent_induction_model: baseline_open_intent_induction_model_fn(
        intent_clustering_model = baseline_intent_clustering_model_fn(
            clustering_algorithm = intent_clustering_model,
            embedding_model = embedding_model.model,
        ),
    ),
    classifier_evaluator: classifier_evaluator,
};

local exp(clustering_model, embedding_model, classifier_evaluator) = 
        baseline_open_intent_induction_experiment(clustering_model.name, clustering_model.model,
         embedding_model, classifier_evaluator);

{
    type: 'meta_experiment',
    run_id:'test' +'_2st',
    datasets: [
        // 'development',
        // 'test-banking',
        // 'test-finance', 
    ],
    experiments:  [
        // exp(clustering_spectral_FTF.spectral, all_mpnet_FTF, logistic_regression_classifier_evaluator_FTF()),
        // exp(clustering_spectral_FTT.spectral, all_mpnet_FTT, logistic_regression_classifier_evaluator_FTT()),
        // exp(clustering_spectral_TTF.spectral, all_mpnet_TTF, logistic_regression_classifier_evaluator_TTF()),
        // exp(clustering_spectral_TTT.spectral, all_mpnet_TTT, logistic_regression_classifier_evaluator_TTT()),

        // exp(clustering_kmeans_FTF.kmeans, all_mpnet_FTF, logistic_regression_classifier_evaluator_FTF()),
        // exp(clustering_kmeans_FTT.kmeans, all_mpnet_FTT, logistic_regression_classifier_evaluator_FTT()),
        // exp(clustering_kmeans_TTF.kmeans, all_mpnet_TTF, logistic_regression_classifier_evaluator_TTF()),
        // exp(clustering_kmeans_TTT.kmeans, all_mpnet_TTT, logistic_regression_classifier_evaluator_TTT()),
        
        //exp(clustering_spectral.spectral, all_mpnet),
        //exp(clustering_kmeans.kmeans, all_mpnet),
        //exp(clustering_gmm.gmm, all_mpnet),
        //exp(clustering_agglomerative.agglomerative, all_mpnet),
    ],
}