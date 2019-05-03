# Active Semi-supervised Variational Autoencoders under Representativeness Constraints
This project applies semi-supervised VAEs on pool-based uncertainty active learning. To respect the underlying class distribution different representativeness strategies can be added.
# Installatiion
active learning code requieres python 3.6+ and tensorflow-1.12+
# Project Structure
    |
    |--data: already preprocessed data of the ADR-Dataset (http://diego.asu.edu/downloads/twitter_annotated_corpus/),
    |         US Airline (https://www.kaggle.com/crowdflower/twitter-airline-sentiment) Sentiment Dataset, 
    |         Large Movie Review dataset (http://ai.stanford.edu/~amaas/data/sentiment/) and
    |         side effect synonyms    
    |--src
        |--java: contains tweet preprocessing (tweet tagger http://www.cs.cmu.edu/~ark/TweetNLP/#%23parser_down and 
        |        Lucene are requiered)
        |--python: 
              |--evaluation: contains proposed active learning methods with differnt models + QBC Naive Bayes with EM
              |       |--asiddhant: adapted code of https://github.com/asiddhant/Active-NLP
              |--pre_processing: Further preprocession files
