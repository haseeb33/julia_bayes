using JSON
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
LDA(numIteration::Int64, M::Int64, topicDirAlpha::Array, wordPolya::Array{Polya}, X::Array{Any,1}, topicPolya::Array{Polya}, Samples::Array{Any}) = LDA(numIteration, M, Dirichlet(topicDirAlpha), wordPolya, X, topicPolya, Samples)

function sample(self::LDA, docs)
    self.X = docs
    D = size(docs)[1]
    
    samples = [[] for i=1:D]
    for d in 1:D
        Nd = size(docs[d])[1]
        samples[d] = [0 for i in 1:Nd]
        push!(self.topicPolya, Polya(self.topicDir))
        
        temp = []
        for i in 1:Nd
            randomSample = rand(1:self.M)
            push!(temp, randomSample)
            addSample(self, d, i, randomSample)
        end
        samples[d] = temp
    end
    self.Samples = samples
    gibbsSampling(self, docs, self.numIteration)
end

function gibbsSampling(self::LDA, docs, numIteration)
    D = size(docs)[1]
    ndMax = Int(floor(maximum([size(doc)[1] for doc in docs])))
    
    C_ = [0 for i in 1:ndMax+1]
    for d in 1:D
        C_[(size(docs[d])[1])+1]+=1 
    end
    samples = self.Samples
    
    for iteration in 1:numIteration
        for d in 1:D
            Nd = size(docs[d])[1]
            for i in 1:Nd
                if samples[d][i] != 0 removeSample(self, d, i, samples[d][i]) end
                samples[d][i] = Sampler_sample(posterior(self, d, i))
                addSample(self, d, i, samples[d][i])
            end
        end
     
        ndkMax = [0 for i in 1:self.M]
        Ck = [[0 for j in 1:ndMax+1] for i in 1:self.M]
        for m in 1:self.M
            for d in 1:D
                ndk = self.topicPolya[d].n[m]
                Ck[m][ndk+1]+=1  
                ndkMax[m] = max(ndkMax[m], ndk)
            end
        end
        dirichlet_optimizeParam(Ck, ndkMax, C_, ndMax+1, 20, self.topicDir)
    end  
    self.Samples = samples
end

function posterior(self::LDA, doc::Int, word::Int)
    v = self.X[doc][word]
    posterior = [0.0 for m in 1:self.M]
    for m in 1:self.M
        posterior[m] = polya_p(m, self.topicPolya[doc]) * polya_p(v, self.wordPolya[m])
    end
    return posterior
end

function addSample(self::LDA, doc::Int, word::Int, m::Int)
    v = self.X[doc][word]
    polya_observe(m, self.topicPolya[doc])
    polya_observe(v, self.wordPolya[m])
end

function removeSample(self::LDA, doc::Int, word::Int, m::Int)
    v = self.X[doc][word]
    polya_forget(m, self.topicPolya[doc])
    polya_forget(v, self.wordPolya[m])
end

function wordPredict(self::LDA, m::Int, v::Int)
    return polya_p(v, self.wordPolya[m])
end

function topicPredict(self::LDA, d::Int, m::Int)
    return polya_p(m, self.topicPolya[d])
end

function topicN(self::LDA, corpus::DocumentSet, topic_n::Int, top_n_words::Int)
    topic_proportion = [];
    for v in 1:size(corpus.reverse_vocabulary)[1]
        prop = wordPredict(self, topic_n, v)
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


function sortedTopDocsForTopics(self::LDA, corpus::DocumentSet)
    topic_distribution = []
    for i in 1:self.M
        topic_dist = []
        for j in 1:corpus.document_size
            push!(topic_dist, topicPredict(self, j, i))
        end
        push!(topic_distribution, topic_dist)
    end
    sorted_topic_distribution = []
    for i in topic_distribution
        push!(sorted_topic_distribution, sortperm(i, rev=true)[1:trunc(Int, (corpus.document_size/self.M))])
    end        
    return sorted_topic_distribution 
end

function topic_of_each_doc(self::LDA, corpus::DocumentSet)
    top_topic_for_each_doc = []
    for i in 1:corpus.document_size
        top_val = 0
        top_topic = 0
        for j in 1:self.M
            v = topicPredict(self, i, j)
            if v>=top_val
                top_val = v
                top_topic = j
            end
        end
        push!(top_topic_for_each_doc, top_topic)
    end
    return top_topic_for_each_doc
end

#------------------------save and load LDA----------------------------
function saveLDA(self::LDA, file)
    lda_dict = Dict()
    lda_dict["numIteration"] = self.numIteration
    lda_dict["M"] = self.M
    lda_dict["topicDir_param"] = self.topicDir.alpha
    lda_dict["wordPolya_n"] = [i.n for i in self.wordPolya]
    lda_dict["X"] = self.X
    lda_dict["topicPolya_n"] = [i.n for i in self.topicPolya]
    lda_dict["Samples"] = self.Samples
    lda_json_string = JSON.json(lda_dict)

    open(file,"w") do f 
        write(f, lda_json_string) 
    end  
end

function loadLDA(file)
    lda1_raw = JSON.parsefile(file);
    numIteration = lda1_raw["numIteration"]
    M = lda1_raw["M"]
    topicDir_alpha = lda1_raw["topicDir_param"]
    wordPolya = [Polya(Dirichlet(length(i), 0.01), i) for i in lda1_raw["wordPolya_n"]]
    X = lda1_raw["X"]
    topicPolya = [Polya(Dirichlet(topicDir_alpha), i) for i in lda1_raw["topicPolya_n"]]
    Samples = lda1_raw["Samples"]
    lda = LDA(numIteration, M, topicDir_alpha, wordPolya, X, topicPolya, Samples)
    return lda
end

#--------------------------------Refinements-------------------------------------

function removeWord(self::LDA, corpus::DocumentSet, word::String, topic::Int)
    word = corpus.vocabulary[word]
    removeWord(self, corpus, word, topic)
end
function removeWord(self::LDA, corpus::DocumentSet, word::Int, topic::Int)
    for doc in enumerate(corpus.documents)
        for w in enumerate(doc[2])
            if w[2]==word
                if self.Samples[doc[1]][w[1]] == topic
                    removeSample(self, doc[1], w[1], self.Samples[doc[1]][w[1]])
                    self.Samples[doc[1]][w[1]] = 0
                end
            end
        end
    end
    param = copy(self.wordPolya[topic].dir.alpha)
    param[word] = 10E-8 # Assign very small prior epsilon 
    self.wordPolya[topic].dir = Dirichlet(param) #Reconstruct the same lda variable
end

function addWord(self::LDA, corpus::DocumentSet, word::String, topic::Int)
    word = corpus.vocabulary[word]
    addWord(self, corpus, word, topic)
end
function addWord(self::LDA, corpus::DocumentSet, word::Int, topic::Int)
    for doc in enumerate(corpus.documents)
        for w in enumerate(doc[2])
            if w[2]==word
                if self.Samples[doc[1]][w[1]]!=topic
                    removeSample(self, doc[1], w[1], self.Samples[doc[1]][w[1]])
                    self.Samples[doc[1]][w[1]] = 0
                end
            end
        end
    end
    param = copy(self.wordPolya[topic].dir.alpha)
    difference = maximum(self.wordPolya[topic].n) - self.wordPolya[topic].n[word] #important discussion part
    param[word] = self.wordPolya[topic].dir.alpha[word] + difference
    self.wordPolya[topic].dir = Dirichlet(param)
end

function addDoc(self::LDA, corpus::DocumentSet, docs, topic::Int)
    if typeof(docs) == Int64
        docs = [docs]
    end
    for doc_idx in docs
        param = copy(self.topicPolya[doc_idx].dir.alpha)
        #previous logic put this was failed on small document dataset(20newsgroup)
        #difference = maximum(self.topicPolya[doc_idx].n) - self.topicPolya[doc_idx].n[topic]
        #param[topic] = self.topicPolya[doc_idx].dir.alpha[topic] + difference
        
        #New logic to improve the effectiveness of this refinement 
        param[topic] = self.topicPolya[doc_idx].dir.alpha[topic] + maximum(self.topicPolya[doc_idx].n)
        for w in enumerate(corpus.documents[doc_idx])
            if self.Samples[doc_idx][w[1]]!=topic
                removeSample(self, doc_idx, w[1], self.Samples[doc_idx][w[1]])
                self.Samples[doc_idx][w[1]] = 0
            end
        end
        self.topicPolya[doc_idx].dir = Dirichlet(param)    
    end          
end

function removeDoc(self::LDA, corpus::DocumentSet, docs, topic::Int)
    if typeof(docs) == Int64
        docs = [docs]
    end
    for doc_idx in docs
        for w in enumerate(corpus.documents[doc_idx])
            removeSample(self, doc_idx, w[1], self.Samples[doc_idx][w[1]])
            self.Samples[doc_idx][w[1]] = 0
        end
        param = copy(self.topicPolya[doc_idx].dir.alpha)
        param[topic] = 10E-8 # Assign very small prior epsilon 
        self.topicPolya[doc_idx].dir = Dirichlet(param) #Reconstruct the same lda variable
    end
end