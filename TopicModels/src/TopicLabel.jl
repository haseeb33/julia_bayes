using PyCall
using Distances
using DataStructures
using ProgressMeter

function label_generation(docs::Array, vocab::Dict, bi_and_tri = [])
    candidate_label_distribution = Dict()
    stopwords = []  
    open("stopwords.txt") do file
        for word in eachline(file)
            push!(stopwords, word)
        end
    end
    nltk = pyimport("nltk")
    for doc_l in docs
        for (pos, word_l) in enumerate(doc_l)
            if word_l in bi_and_tri || bi_and_tri == []
                if TopicModels.preprocess_lemma(word_l, stopwords, nltk) != "abcxyz"
                    if haskey(candidate_label_distribution, word_l)
                        each_word_frequency(pos, doc_l, candidate_label_distribution, vocab, stopwords, nltk)
                    else
                        candidate_label_distribution[word_l] = Dict()
                        each_word_frequency(pos, doc_l, candidate_label_distribution, vocab, stopwords, nltk)
                    end
                end
            end
        end
    end
    return candidate_label_distribution
end

function each_word_frequency(pos, doc, candidate_label_distribution, vocab, stopwords, nltk, window_size=50/2)
    backward_count = 0; forward_count = 0
    for i in pos+1:length(doc)
        if TopicModels.preprocess_lemma(doc[i], stopwords, nltk) != "abcxyz" && doc[i] in keys(vocab)
            if haskey(candidate_label_distribution[doc[pos]], doc[i])
                candidate_label_distribution[doc[pos]][doc[i]] +=1
            else
                candidate_label_distribution[doc[pos]][doc[i]] = 1
            end
            forward_count+=1
        end
        if forward_count == window_size
            break
        end
    end
    for i in pos-1:-1:1
        if TopicModels.preprocess_lemma(doc[i], stopwords, nltk) != "abcxyz" && doc[i] in keys(vocab)
            if haskey(candidate_label_distribution[doc[pos]], doc[i])
                candidate_label_distribution[doc[pos]][doc[i]] +=1
            else
                candidate_label_distribution[doc[pos]][doc[i]] = 1
            end
            backward_count +=1
        end
        if backward_count == window_size
            break
        end
    end
end


#change based on paper (not cosine similarity but KL divergence)
# check more words on topic models (all words in vocab set) (14707 for this case)
# For now take three topic labels(we can change bases on results)
# optional discriminstive labeling  (Depends on the results).
function label_ranking(candidate_label_distribution, topics, proportions, top=3, base_count_kl = 0.01)
    top_labels = Dict()
    @showprogress 1 "Computing..." for (label, label_dist) in candidate_label_distribution
        freq_dict = DefaultDict(0, label_dist)
        for (index, topic) in enumerate(topics)
            label_distribution = [(freq_dict[t]+base_count_kl)/(sum(values(label_dist))+(length(topic)*base_count_kl)) for t in topic]
            #print(label_distribution)
            distance_v = TopicModels.kl_divergence_fn(proportions[index], label_distribution)
            #print(distance_v)
            if index in keys(top_labels)
                for (pos, top_label) in enumerate(top_labels[index])
                    if distance_v < top_label[2] && !(label in [l[1] for l in top_labels[index]]) 
                        insert!(top_labels[index], pos, [label, distance_v, label_distribution])
                        if length(top_labels[index]) > top
                            pop!(top_labels[index])
                        end
                    end
                end 
            else
                top_labels[index] = [[label, distance_v, label_distribution]]
            end 
        end   
    end
    return top_labels
end

function kl_divergence_fn(P, Q)
    sum = 0
    for (pos, x) in enumerate(P)
        if x !=0
            sum += x* log(x/Q[pos])
        end
    end
    return sum
end