#!/usr/bin/env python

import multiprocessing
from typing import List, Union, Dict
from pathlib import Path

import gensim
import numpy as np
from gensim.models import Word2Vec

import constants as c
from tweet import Tweet


class WordEmbeddingModel:
    w2v_model: Word2Vec

    def __init__(self, latent_space_dim: int = c.LATENT_SPACE_DIM, window_size: int = c.WINDOW_SIZE, save_file_path: Path = c.W2V_SAVE_FILE_NAME, min_count: int = 5):
        self.latent_space_dim = latent_space_dim
        self.window_size = window_size
        self.save_file_path = save_file_path
        self.min_count = min_count

        cores = multiprocessing.cpu_count()
        self.w2v_model = Word2Vec(min_count=self.min_count, window=self.window_size, size=self.latent_space_dim, sample=1e-5, alpha=0.01,
                            min_alpha=0.001, negative=10,
                            workers=cores - 1)

    def train(self, tweet_corpus: List[Tweet]):
        """ Create and stores Word2Vec model """
        tweet_texts = [tweet.text for tweet in tweet_corpus]
        self.w2v_model.build_vocab(tweet_texts, progress_per=10)
        self.w2v_model.train(tweet_texts, total_examples=self.w2v_model.corpus_count, epochs=50, report_delay=1)
        self.w2v_model.save(self.save_file_path)
    
    def load(self):
        self.w2v_model = Word2Vec.load(c.W2V_SAVE_FILE_NAME)

    def remove_hashtags_not_part_of_the_vocab(self, hashtags: List[str]) -> List[str]:
        remaining_hashtags = [hashtag for hashtag in hashtags if hashtag in self.w2v_model.wv.vocab]

        if hashtags != remaining_hashtags:
            print(
                'some hashtags were filtered out since they are not contained in the Word2Vec-Vocab',
                f'tweet-hashtags: {hashtags} | hashtags after filtering: {remaining_hashtags}'
            )

        return remaining_hashtags

    def generate_target(self, hashtags: List[str]) -> Union[np.array, None]:
        hashtag_embeddings = [self.w2v_model.wv[hashtag] for hashtag in hashtags]
        return np.mean(hashtag_embeddings) if len(hashtag_embeddings) > 0 else None

def retain_hashtags(top_emb): # TODO rewrite?
    top_emb_hts = []
    for tuple in top_emb:
        if '#' in tuple[0]:
            if '_' in tuple[0]:  # bigrams
                split = tuple[0].split(sep='_')
                for h in split:
                    if '#' in h:
                        top_emb_hts.append((h, tuple[1]))
            else:
                top_emb_hts.append(tuple)
    return top_emb_hts
