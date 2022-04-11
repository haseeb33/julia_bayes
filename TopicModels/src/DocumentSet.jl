mutable struct DocumentSet
    docs_text::Array
    documents::Array
    document_size::Int
    vocab_count::Int
    vocabulary::Dict{}
    reverse_vocabulary::Array
end

DocumentSet() = DocumentSet([], [], 0, 0, Dict(), [])

function readData(path::String)
    corpus = DocumentSet()
    open(path) do file
        for doc in eachline(file)
            addDocument(corpus, doc)
        end
    end
    return corpus
end

function readData(texts::Array)
    corpus = DocumentSet()
    for doc in texts
        addDocument(corpus, doc)
    end
    return corpus
end

function addDocument(self::DocumentSet, line::String)
    words = split(line)
    addDocument(self, words)
end

function addDocument(self::DocumentSet, words::Array)
    if isempty(words)
        return nothing
    end
    codes = []
    for i in words
        if haskey(self.vocabulary, i)
            push!(codes, self.vocabulary[i])
        else 
            self.vocab_count+=1
            self.vocabulary[i] = self.vocab_count
            push!(self.reverse_vocabulary, i)
            push!(codes, self.vocab_count)
        end
    end
    push!(self.documents, codes)
    push!(self.docs_text, words)
    self.document_size+=1
end

function transform(self::DocumentSet, line::String)
    words = split(line)
    codes = []
    for i in words
        code = get(self.vocabulary, i, -1)
        if code != -1
            push!(codes, code)
        end  
    end
    return sort(codes)
end

function sampleDocuments(self::DocumentSet, numDocs::Int)
    subD = sample(self.documents, numDocs, replace = false)
    return subD  
end

function OnlinesampleDocuments(self::DocumentSet, numDocs::Int, iter::Int)
    subD = self.documents[(iter-1)*numDocs+1: (iter)*numDocs]
    return subD
end

function getTermFreq(self::DocumentSet)
    tf = [0 for i=1:self.vocab_count]
    for doc in self.documents
        for w in doc
            tf[w]+=1 
        end
    end
    return tf
end