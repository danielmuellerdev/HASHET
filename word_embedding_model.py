#!/usr/bin/env python

import multiprocessing
from typing import List
from pathlib import Path

import gensim
import numpy as np
from gensim.models import Word2Vec

import constants as c


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

    def train(self, tweet_corpus: List[List[str]]):
        """ Create and stores Word2Vec model """
        self.w2v_model.build_vocab(tweet_corpus, progress_per=10)
        self.w2v_model.train(tweet_corpus, total_examples=self.w2v_model.corpus_count, epochs=50, report_delay=1)
        self.w2v_model.save(self.save_file_path)
    
    def load(self):
        self.w2v_model = Word2Vec.load(c.W2V_SAVE_FILE_NAME)

    def generate_target(self, tweet):
        tw_list = tweet.split()
        den = 0
        sent_embedding = np.zeros(self.latent_space_dim)
        for word in tw_list:
            try:
                emb = self.w2v_model.wv[word]
                den += 1
                sent_embedding += emb
            except:
                pass
        return None if den == 0 else sent_embedding / den      

def retain_hashtags(top_emb):
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
