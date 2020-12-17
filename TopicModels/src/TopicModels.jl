module TopicModels

include("DocumentSet.jl")
export DocumentSet
export documentset_readData, documentset_addDocument, documentset_transform, documentset_sampleDocuments
export documentset_OnlinesampleDocuments, documentset_getTermFreq

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
export lda_posterior, lda_addSample, lda_removeSample, lda_wordPredict, lda_topicPredict, lda_sample, lda_topicN

end # module
