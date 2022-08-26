# Hashtag-Embedding Comparison
Code used for comparing two types of Hashtag-Embeddings:
* Word2Vec
* Word2vec mapped (to the sentence space of tweets) using a trained 2-layer MLP

## Hashtag-to-Sentence-Space Mapper
### Model 
* A simple 2-layer MLP

### Data
* 4000 unique Hashtags used. Train / Val / Test split: 2800 / 400 / 800
* sample:
   * x: `Word2Vec(hashtag)`
   * y: `mean([SentenceEmbedding(tweet.text) for all tweets using the hashtag])` = centroid in the sentence embedding space. SentenceEmbedding: generated using a pretrained BERT model
* 200,000 Tweets used for creating the training data (dataset not uploaded)

### Training Objective
* Minimize the cosine-distance-loss between the predicted centroid and the actual centroid in the sentence embedding space for a given hashtag

## HASHET
* Dataset creation and mapping procedure were inspired by the forked HASHET model
