module TopicModels
using CSV, DataFrames, PyCall, JSON

include("DocumentSet.jl")
export DocumentSet
export readData, addDocument, transform, sampleDocuments
export OnlinesampleDocuments, getTermFreq

include("Dirichlet.jl")
export Dirichlet
export dirichlet_optimizeParam, dirichlet_set

include("Polya.jl")
export Polya
export polya_p, polya_observe, polya_forget, polya_getCount

include("Sampler.jl")
export Sampler_sample

include("LDA.jl")
export LDA
export posterior, addSample, removeSample, wordPredict, topicPredict, sample, topicN, gibbsSampling
export removeWord, addWord,removeDoc, saveLDA, loadLDA

include("TopicLabel.jl")
export label_generation, each_word_frequency, label_ranking, kl_divergence

include("APIs.jl")
export preprocess, preprocess_lemma, train, show_topics, apply_refinement, topic_coherence, train_phrase_model

include("Keyphrase.jl")
export Keyphrase
export load_keyphrase, top_keyphrases_of_topic, similarity_of_keyphrases, keyphrase_cluster

end # module
