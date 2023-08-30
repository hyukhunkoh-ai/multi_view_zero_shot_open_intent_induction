local default_settings = {
    class_weight: 'balanced'
};

local logistic_regression_classifier_evaluator(settings = default_settings) = {
    type: 'logistic_regression_classifier_evaluator',
    sentence_encoder: import '../embedding-model/sms.jsonnet',
    classifier_settings: settings
};

logistic_regression_classifier_evaluator