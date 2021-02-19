mutable struct LDA
    numIteration::Int64
    M::Int64
    topicDir::Dirichlet
    wordPolya::Array{Polya}
    X::Array{Any,1}
    topicPolya::Array{Polya}
    Samples::Array{Any}
end

LDA(topicPrior::Dirichlet, wordPrior::Dirichlet) = LDA(100, topicPrior.K, topicPrior, [Polya(wordPrior) for i in 1:topicPrior.K], [[]], [], [[]])

function lda_sample(docs, lda_obj::LDA)
    lda_obj.X = docs
    D = size(docs)[1]
    
    samples = [[] for i=1:D]
    for d in 1:D
        Nd = size(docs[d])[1]
        samples[d] = [0 for i in 1:Nd]
        push!(lda_obj.topicPolya, Polya(lda_obj.topicDir))
        
        temp = []
        for i in 1:Nd
            randomSample = rand(1:lda_obj.M)
            push!(temp, randomSample)
            lda_addSample(d, i, randomSample, lda_obj)
        end
        samples[d] = temp
    end
    lda_obj.Samples = samples
    lda_gibbsSampling(docs, lda_obj.numIteration, lda_obj)
end

function lda_gibbsSampling(docs, numIteration, lda_obj::LDA)
    D = size(docs)[1]
    ndMax = Int(floor(maximum([size(doc)[1] for doc in docs])))
    
    C_ = [0 for i in 1:ndMax+1]
    for d in 1:D
        C_[(size(docs[d])[1])+1]+=1 
    end
    samples = lda_obj.Samples
    
    for iteration in 1:numIteration
        for d in 1:D
            Nd = size(docs[d])[1]
            for i in 1:Nd
                if samples[d][i] != 0 lda_removeSample(d, i, samples[d][i], lda_obj) end
                samples[d][i] = Sampler_sample(lda_posterior(d, i, lda_obj))
                lda_addSample(d, i, samples[d][i], lda_obj)
            end
        end
     
        ndkMax = [0 for i in 1:lda_obj.M]
        Ck = [[0 for j in 1:ndMax+1] for i in 1:lda_obj.M]
        for m in 1:lda_obj.M
            for d in 1:D
                ndk = lda_obj.topicPolya[d].n[m]
                Ck[m][ndk+1]+=1  
                ndkMax[m] = max(ndkMax[m], ndk)
            end
        end
        dirichlet_optimizeParam(Ck, ndkMax, C_, ndMax+1, 20, lda_obj.topicDir)
    end  
    lda_obj.Samples = samples
end

function lda_posterior(d::Int, i::Int, lda_obj::LDA)
    v = lda_obj.X[d][i]
    posterior = [0.0 for i in 1:lda_obj.M]
    for m in 1:lda_obj.M
        posterior[m] = polya_p(m, lda_obj.topicPolya[d]) * polya_p(v, lda_obj.wordPolya[m])
    end
    return posterior
end

function lda_addSample(d::Int, i::Int, m::Int, lda_obj::LDA)
    v = lda_obj.X[d][i]
    polya_observe(m, lda_obj.topicPolya[d])
    polya_observe(v, lda_obj.wordPolya[m])
end

function lda_removeSample(d::Int, i::Int, m::Int, lda_obj::LDA)
    v = lda_obj.X[d][i]
    polya_forget(m, lda_obj.topicPolya[d])
    polya_forget(v, lda_obj.wordPolya[m])
end

function lda_wordPredict(m::Int, v::Int, lda_obj::LDA)
    return polya_p(v, lda_obj.wordPolya[m])
end

function lda_topicPredict(d::Int, m::Int, lda_obj::LDA)
    return polya_p(m, lda_obj.topicPolya[d])
end

function lda_topicN(topic_n::Int, top_n_words::Int, corpus::DocumentSet, lda::LDA)
    topic_proportion = [];
    for v in 1:size(corpus.reverse_vocabulary)[1]
        prop = TopicModels.lda_wordPredict(topic_n, v, lda)
        push!(topic_proportion, prop)
    end
    topic_words_idx = sortperm(topic_proportion, rev=true)[1:top_n_words]
    top_words = []
    for i in topic_words_idx
        wrd = corpus.reverse_vocabulary[i]
        push!(top_words, wrd)
    end
    topic_proportion = topic_proportion[topic_words_idx]
    return top_words, [round(i, digits=5) for i in topic_proportion]
end

function lda_removeWord(word::String, topic::Int, corpus::DocumentSet, lda_obj::LDA)
    word = corpus.vocabulary[word]
    lda_removeWord(word, topic, corpus, lda_obj)
end
function lda_removeWord(word::Int, topic::Int, corpus::DocumentSet, lda_obj::LDA)
    for doc in enumerate(corpus.documents)
        for w in enumerate(doc[2])
            if w[2]==word
                if lda_obj.Samples[doc[1]][w[1]] == topic
                    lda_removeSample(doc[1], w[1], lda_obj.Samples[doc[1]][w[1]], lda_obj)
                    lda_obj.Samples[doc[1]][w[1]] = 0
                end
            end
        end
    end
    param = copy(lda_obj.wordPolya[topic].dir.alpha)
    param[word] = 10E-8 # Assign very small prior epsilone 
    lda_obj.wordPolya[topic].dir = Dirichlet(param) #Reconstruct the same lda variable
end

function lda_addWord(word::String, topic::Int, corpus::DocumentSet, lda_obj::LDA)
    word = corpus.vocabulary[word]
    lda_addWord(word, topic, corpus, lda_obj)
end
function lda_addWord(word::Int, topic::Int, corpus::DocumentSet, lda_obj::LDA)
    for doc in enumerate(corpus.documents)
        for w in enumerate(doc[2])
            if w[2]==word
                if lda_obj.Samples[doc[1]][w[1]]!=topic
                    lda_removeSample(doc[1], w[1], lda_obj.Samples[doc[1]][w[1]], lda_obj)
                    lda_obj.Samples[doc[1]][w[1]] = 0
                end
            end
        end
    end
    param = copy(lda_obj.wordPolya[topic].dir.alpha)
    difference = maximum(lda_obj.wordPolya[topic].n) - lda_obj.wordPolya[topic].n[word] #important discussion part
    param[word] = lda_obj.wordPolya[topic].dir.alpha[word] + difference
    lda_obj.wordPolya[topic].dir = Dirichlet(param)
end