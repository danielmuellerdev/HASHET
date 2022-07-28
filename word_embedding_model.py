import multiprocessing
from typing import List

from gensim.models import Word2Vec
from torch import Tensor
import torch

from tweet import Tweet


class WordEmbeddingModel:
    def __init__(self, latent_space_dim: int = 150, window_size: int = 5, min_count: int = 5):
        self.latent_space_dim = latent_space_dim
        self.window_size = window_size
        self.min_count = min_count

        cores = multiprocessing.cpu_count()
        self._w2v_model = Word2Vec(min_count=self.min_count, window=self.window_size, size=self.latent_space_dim, sample=1e-5, alpha=0.01,
                            min_alpha=0.001, negative=10,
                            workers=cores - 1)

    def train(self, tweets: List[Tweet]):
        sentences = [tweet.tokens for tweet in tweets]
        self._w2v_model.build_vocab(sentences)
        self._w2v_model.train(sentences, total_examples=self._w2v_model.corpus_count, epochs=50, report_delay=1)

    @property
    def vocab(self):
        return self._w2v_model.wv.vocab

    def get_embedding(self, word: str) -> Tensor:
        return torch.tensor(self._w2v_model.wv[word]) if word in self._w2v_model.wv.vocab else None

