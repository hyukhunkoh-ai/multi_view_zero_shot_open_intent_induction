{
    type: 'caching_sentence_embedding_model',
    sentence_embedding_model: {
        type: 'sentence_transformers_model',
        combination: 'fft',
    },
    cache_path: 'cache',
    prefix: 'FFT',
}