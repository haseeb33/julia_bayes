mutable struct DocumentSet
    documents::Array
    document_size::Int
    vocab_count::Int
    vocabulary::Dict{}
    reverse_vocabulary::Array
end

DocumentSet() = DocumentSet([],0, 0, Dict(), [])

function documentset_readData(path::String)
    corpus = DocumentSet()
    open(path) do file
        for doc in eachline(file)
            documentset_addDocument(doc, corpus)
        end
    end
    return corpus
end

function documentset_readData(texts::Array)
    corpus = DocumentSet()
    for doc in texts
        documentset_addDocument(doc, corpus)
    end
    return corpus
end

function documentset_addDocument(line::String, documentset_obj::DocumentSet)
    if isempty(line)
        return nothing
    end
    words = split(line)
    codes = []
    for i in words
        if haskey(documentset_obj.vocabulary, i)
            push!(codes, documentset_obj.vocabulary[i])
        else 
            documentset_obj.vocab_count+=1
            documentset_obj.vocabulary[i] = documentset_obj.vocab_count
            push!(documentset_obj.reverse_vocabulary, i)
            push!(codes, documentset_obj.vocab_count)
        end
    end
    push!(documentset_obj.documents, codes)
    documentset_obj.document_size+=1
end

function documentset_addDocument(words::Array, documentset_obj::DocumentSet)
    if isempty(words)
        return nothing
    end
    codes = []
    for i in words
        if haskey(documentset_obj.vocabulary, i)
            push!(codes, documentset_obj.vocabulary[i])
        else 
            documentset_obj.vocab_count+=1
            documentset_obj.vocabulary[i] = documentset_obj.vocab_count
            push!(documentset_obj.reverse_vocabulary, i)
            push!(codes, documentset_obj.vocab_count)
        end
    end
    push!(documentset_obj.documents, codes)
    documentset_obj.document_size+=1
end

function documentset_transform(line::String, documentset_obj::DocumentSet)
    words = split(line)
    codes = []
    for i in words
        code = get(documentset_obj.vocabulary, i, -1)
        if code != -1
            push!(codes, code)
        end  
    end
    return sort(codes)
end

function documentset_sampleDocuments(numDocs::Int, documentset_obj::DocumentSet)
    subD = sample(documentset_obj.documents, numDocs, replace = false)
    return subD  
end

function documentset_OnlinesampleDocuments(numDocs::Int, iter::Int, documentset_obj::DocumentSet)
    subD = documentset_obj.documents[(iter-1)*numDocs+1: (iter)*numDocs]
    return subD
end

function documentset_getTermFreq(documentset_obj::DocumentSet)
    tf = [0 for i=1:documentset_obj.vocab_count]
    for doc in documentset_obj.documents
        for w in doc
            tf[w]+=1 
        end
    end
    return tf
end