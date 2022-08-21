using PyCall
using JSON
using CSV, DataFrames

function preprocess(path::String, use="all")
    
    papers = CSV.read(path, DataFrame);
    docs = papers.paper_text;
    #titles = papers.title; modifiy implementation to store titles in document object
    if typeof(use) == Int64
        docs = docs[length(docs)-use+1:length(docs)]
    end
    specialchars = ['!', '”', '#', '$', '%', '&', '’', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '>', '=', '@', '?', '[', ']', '^', '_', '{', '}', '|', '~']
    stopwords = []
    open("stopwords.txt") do file
        for word in eachline(file)
            push!(stopwords, word)
        end
    end
    
    #Remove stop words, special characters, numbers, and 2_or_less char words and lemmatization
    nltk = pyimport("nltk")
    new_docs = []
    no_lemma_docs = []
    
    for line in docs
        doc = split(line)
        temp = []; no_lemma_temp = []
        #p_of_s = nltk.pos_tag(doc) Not using because of mismatch with candidate label docs
        for word in doc
            #word = word_tag[1][1]; tag = word_tag[1][2]
            word = lowercase(replace.(word, specialchars => ""))
            if all(c -> 'a' <= c <= 'z' || 'A' <= c <= 'Z', word)
                push!(no_lemma_temp, word)
                #Lemma function
                if preprocess_lemma(word, stopwords, nltk) != "abcxyz"
                    push!(temp, word)
                end
            end
        end
        push!(new_docs, temp)
        push!(no_lemma_docs, no_lemma_temp)
    end
    
    return TopicModels.readData(new_docs), no_lemma_docs
end

function preprocess_lemma(word, stopwords, nltk)
    if !(word in stopwords) 
        if length(word) > 2 && tryparse(Float64, word) == nothing
            lemmatizer = nltk.stem.WordNetLemmatizer()
            word = lemmatizer.lemmatize(word) #pos=tag not using because of complications
            return word
        end
    end
    return "abcxyz"
end

function train(corpus::DocumentSet, numTopics::Int)
    wordPrior = TopicModels.Dirichlet(corpus.vocab_count, 0.01)
    M = numTopics
    alpha = [0.01 for i in 1:M]
    topicPrior = TopicModels.Dirichlet(alpha)
    lda = TopicModels.LDA(topicPrior, wordPrior)
    TopicModels.sample(lda, corpus.documents)
    return lda
end

function show_topics(lda::LDA, corpus::DocumentSet, top=10, show=true)
    topics = []; proportions = []
    for m in 1:lda.M
        words, prop = TopicModels.topicN(lda, corpus, m, top)
        push!(topics, words)
        push!(proportions, prop)
        if show
            println(words)
            #println(prop)
            println("----------------------")
        end
    end
    return topics, proportions
end

function apply_refinement(lda::LDA, corpus::DocumentSet, refi::String, word_or_doc_or_kp, topic::Int, kp=nothing)
    if refi in ["Add", "add", "a"]
        TopicModels.addWord(lda, corpus, word_or_doc_or_kp, topic)
        TopicModels.gibbsSampling(lda, corpus.documents, 20);
    elseif refi in ["Remove", "remove", "r"]
        TopicModels.removeWord(lda, corpus, word_or_doc_or_kp, topic)
        TopicModels.gibbsSampling(lda, corpus.documents, 20);
    elseif refi in ["Remove_doc", "remove_doc", "remove doc", "Remove doc", "R_D", "r_d"]
        TopicModels.removeDoc(lda, corpus, word_or_doc_or_kp, topic)
        TopicModels.gibbsSampling(lda, corpus.documents, 20);
    elseif refi in ["Remove_kp", "remove_kp", "R_kp", "r_kp"]
        #Not complete yet,
        topic_distributions = TopicModels.sortedTopDocsForTopics(lda, corpus);
        cluster_kp, docs_have = TopicModels.top_x_kp_of_topic_m(kp, topic_distributions, 5, topic);
        apply_refinement(lda, corpus, "R_D", docs_have[word_or_doc_or_kp], topic);

    else
        Println("It is not a valid refinement")
    end
end

#--------------------------Gensim Related, coherence, phrase_mode -----------------------------

function topic_coherence(corpus::DocumentSet, topics::Array, wiki::Bool=false)
    corpora = pyimport("gensim.corpora")
    cm = pyimport("gensim.models.coherencemodel")
    if wiki
        id2word = corpora.Dictionary.load_from_text("data_wordids.txt.bz2")
        mm = corpora.MmCorpus("data_tfidf.mm")
        new_topics = []
        for i in topics
            temp = []
            for j in i
                try
                    id2word.token2id[j]
                    push!(temp, j)
                catch;
                end
            end
            push!(new_topics, temp)
        end
    coherence_model = cm.CoherenceModel(topics=new_topics, corpus=mm, dictionary=id2word, coherence="u_mass")
    else
        BoW = [[(w, Base.count(i->(i==w), doc)) for w in collect(Set(doc))] for doc in corpus.documents];
        dct = corpora.Dictionary.from_corpus(BoW, id2word=Dict((index-1, value) for (index, value) in enumerate(corpus.reverse_vocabulary)))
        coherence_model = cm.CoherenceModel(topics=new_topics, corpus=BoW, dictionary=dct, coherence="u_mass")
    end
    coherence = coherence_model.get_coherence()
    return coherence
end

function train_phrase_model(docs::Array, vocab::Dict, min_count=10, bi_tri=true)
    bi_and_tri = []
    pharases = pyimport("gensim.models.phrases")
    bi_model = pharases.Phrases(docs, min_count=10, threshold=1, connector_words=pharases.ENGLISH_CONNECTOR_WORDS)
    bi_docs = bi_model.__getitem__(docs)
    if bi_tri
        for (phrase, score) in bi_model.find_phrases(docs)
            push!(bi_and_tri, phrase)
        end
    end
    tri_model = pharases.Phrases(bi_docs, min_count=10, threshold=1, connector_words=pharases.ENGLISH_CONNECTOR_WORDS)
    if bi_tri
        for (phrase, score) in tri_model.find_phrases(bi_docs)
            push!(bi_and_tri, phrase)
        end
    end
    tri_docs = tri_model.__getitem__(bi_docs)
    tri_docs = [doc for doc in tri_docs]
    candidate_label_distribution = TopicModels.label_generation(tri_docs, vocab, bi_and_tri)
    return candidate_label_distribution
end