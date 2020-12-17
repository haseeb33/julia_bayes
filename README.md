# Bayesian models

LDA using collapsed gibbs sampling is implemented.  

To use the LDA as a module, please use the ```TopicModels``` julia package.

Steps to use TopicModels package:

    git clone #clone the repo
    cd handson_julia  #cd to parent directory of TopicModels folder
    Pkg.activate("TopicModels") #activate the package in julia REPL
    corpus = TopicModels.documentset_readData("news.txt") #import the document collection
    wordPrior = TopicModels.Dirichlet(corpus.vocab_count, 0.01) #dirichlet word prior
    M = 3                             #number of topics
    alpha = [0.01 for i in 1:M]      
    topicPrior = TopicModels.Dirichlet(alpha); #dirichlet topic prior
    lda = TopicModels.LDA(topicPrior, wordPrior) #build LDA struct using word prior and topic prior
    samples = TopicModels.lda_sample(corpus.documents, lda) #run LDA with collapsed gibbs sampling
    words, proportions = TopicModels.lda_topicN(1, 10, corpus, lda) #top 10 words and topic proportions of Topic 1
    
LDA on dummy news dataset as well as on NIPS paper dataset is applied. Implementation and results can be viewed in ```LDA_with_package.ipynb``` notebook.

To view the complete implementation in jupyter notebook, please have a look at ```try_LDA.ipynb```
