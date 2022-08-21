using JSON
using LSHFunctions, LinearAlgebra

mutable struct Keyphrase
    totalKeyphrases::Int64
    #uniqueKeyphrases::Int64
    allKeyphrases::Array
    #documentwiseKeyphrases::Array
    keyphraseEmbedding::Dict{}
    keyphraseSimilarity::Dict{}
    keyphrasesOnly::Array
    
end

Keyphrase() = Keyphrase(0, [], Dict(), Dict(), [])

function load_keyphrase(fileKPDoc, fileEMB, fileSim, fileKP)
    self = Keyphrase()
    keyphraseLines = readlines(fileKPDoc)
    predData = [JSON.parse(l) for l in keyphraseLines]
    for i in predData
        push!(self.allKeyphrases, i["pred"])
        self.totalKeyphrases += length(i["pred"])
    end 
    self.keyphraseEmbedding = JSON.parsefile(fileEMB);
    self.keyphraseSimilarity = JSON.parsefile(fileSim);
    self.keyphrasesOnly = JSON.parsefile(fileKP);
    return self
end
    
function keyphrases_of_topic(self::Keyphrase, topicsDistribution, id)
    keyphrases = Dict()
    keyphrasesCount = 0
    documentWiseKeyphrases = []
    for i in topicsDistribution[id]
        push!(documentWiseKeyphrases, self.allKeyphrases[i])
        for j in self.allKeyphrases[i]
            keyphrasesCount +=1
            if ! haskey(keyphrases, j)
                keyphrases[j] = 1
            else
                keyphrases[j] = keyphrases[j]+1
            end
        end
    end
    #println("Total keyphrases are: ", keyphrasesCount)
    #println("Unique keyphrases are: ", length(keyphrases))
    return(sort(collect(keyphrases), by = x->x[2], rev=true), documentWiseKeyphrases)
end

function similarity_of_keyphrases(self::Keyphrase, all_kp, np)
    similarity_matrix = np.empty((length(all_kp), length(all_kp)))
    for (i_idx, i) in enumerate(all_kp)
        for (j_idx, j) in enumerate(all_kp)
            s = cossim(self.keyphraseEmbedding[i.first], self.keyphraseEmbedding[j.first])
            similarity_matrix[i_idx, j_idx] = s
        end
    end
    return similarity_matrix
end

function keyphrase_cluster(self::Keyphrase, topic_doc_kp, max_kp_count=5, sim_threshold=0.85)
    kp_topic_ls = [i.first for i in topic_doc_kp];
    cluster_kp = []; all_cluster_kps = []; kp_c = 0
    for (i_idx, i) in enumerate(kp_topic_ls)
        if !(i in all_cluster_kps)
            temp_kp_ls = self.keyphraseSimilarity[i]
            temp_dict = Dict(); temp_dict[i] = []
            for j in findall(x->x==1, ifelse.(temp_kp_ls.>sim_threshold, true, false))
                if self.keyphrasesOnly[j] in kp_topic_ls
                    push!(temp_dict[i], self.keyphrasesOnly[j])
                    push!(all_cluster_kps, self.keyphrasesOnly[j])
                end
            end
            push!(cluster_kp, temp_dict)
            kp_c+=1
        end
        if kp_c>=max_kp_count
            break
        end
    end
    return cluster_kp
end

function top_x_kp_of_topic_m(self::Keyphrase, topic_distributions, x, m, sim_threshold=0.85)
    all_kp, documentwise_kp_ls = keyphrases_of_topic(self, topic_distributions, m);
    cluster_kp = keyphrase_cluster(self, all_kp, x, sim_threshold);
    docs_have = []
    for i in cluster_kp
        temp_docs_have = []
        for j in collect(values(i))[1]
            for (k_idx, k) in enumerate(documentwise_kp_ls)
                if j in k && !(topic_distributions[m][k_idx] in temp_docs_have)
                    push!(temp_docs_have, topic_distributions[m][k_idx])
                end
            end
        end
        push!(docs_have, copy(temp_docs_have))
    end
    return (cluster_kp, docs_have)
end

function top_x_kp_of_topic_m_for_all_docs(self::Keyphrase, topic_distributions, x, m, sim_threshold=0.85)
    all_kp, documentwise_kp_ls = keyphrases_of_topic(self, topic_distributions, m);
    cluster_kp = keyphrase_cluster(self, all_kp, x, sim_threshold);
    docs_have = []
    for i in cluster_kp
        temp_docs_have = []
        for j in collect(values(i))[1]
            for (k_idx, k) in enumerate(self.allKeyphrases)
                if j in k && !(k_idx in temp_docs_have)
                    push!(temp_docs_have, k_idx)
                end
            end
        end
        push!(docs_have, copy(temp_docs_have))
    end
    return (cluster_kp, docs_have)
end