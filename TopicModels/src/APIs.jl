using CSV, DataFrames

function preprocess(path::String, use="all")
    
    papers = CSV.read("papers.csv", DataFrame);
    docs = papers.paper_text;
    stopwords = []
    specialchars = ['!', '”', '#', '$', '%', '&', '’', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '>', '=', '@', '?', '[', ']', '^', '_', '{', '}', '|', '~']
    open("stopwords.txt") do file
        for word in eachline(file)
            push!(stopwords, word)
        end
    end
    
    #Remove stop words, special characters, numbers, and 2_or_less char words
    new_docs = []
    for line in docs
        doc = split(line)
        temp = []
        for word in doc
            word = lowercase(replace.(word, specialchars => ""))
            if !(word in stopwords) 
                if length(word) > 2 && tryparse(Float64, word) == nothing
                    push!(temp, word)
                end
            end
        end
        push!(new_docs, temp)
    end
    
    if typeof(use) == Int64
        return TopicModels.documentset_readData(new_docs[length(new_docs)-use:length(new_docs)]);
    end
    return TopicModels.documentset_readData(new_docs);
end

function train(corpus::DocumentSet, numTopics::Int)
    wordPrior = TopicModels.Dirichlet(corpus.vocab_count, 0.01)
    M = numTopics
    alpha = [0.01 for i in 1:M]
    topicPrior = TopicModels.Dirichlet(alpha)
    lda_obj = TopicModels.LDA(topicPrior, wordPrior)
    TopicModels.lda_sample(corpus.documents, lda_obj)
    return lda_obj
end

function show_topics(corpus::DocumentSet, lda_obj::LDA, top=10)
    for m in 1:lda_obj.M
        words, proportions = TopicModels.lda_topicN(m, top, corpus, lda_obj)
        println(words)
        println(proportions)
        println("----------------------")
    end
end

function apply_refinement(corpus::DocumentSet, lda_obj::LDA, refi::String, word, topic::Int)
    if refi in ["Add", "add", "a"]
        TopicModels.lda_addWord(word, topic, corpus, lda_obj)
        TopicModels.lda_gibbsSampling(corpus.documents, 20, lda_obj);
    elseif refi in ["Remove", "remove", "r"]
        TopicModels.lda_removeWord(word, topic, corpus, lda_obj)
        TopicModels.lda_gibbsSampling(corpus.documents, 20, lda_obj);
    end
end