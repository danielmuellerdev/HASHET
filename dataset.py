from cgi import test
from collections import defaultdict
from copyreg import pickle
from enum import unique
from typing import List, Tuple
from pathlib import Path
import pickle
import random
from pyparsing import Word

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from tweet import Tweet
from word_embedding_model import WordEmbeddingModel
from sentence_embedding_model import SentenceEmbeddingModel


class CachedDataset(torch.utils.data.Dataset):
    def __init__(
        self, hashtags: List[str], tweets: List[Tweet], word_emb_model: WordEmbeddingModel,
        sent_emb_model: SentenceEmbeddingModel
    ):
        self.hashtags = hashtags
        self.word_emb_model = word_emb_model
        self.sent_emb_model = sent_emb_model

        self._hashtag_to_tweets = defaultdict(list)
        for tweet in tweets:
            for hashtag in tweet.hashtags:
                self._hashtag_to_tweets[hashtag].append(tweet)

        self._cache = {}

    def _get_tweets_by_hashtag(self, hashtag: str) -> List[Tweet]:
        return self._hashtag_to_tweets[hashtag]

    def cache_all_and_store(self, file_path: Path, report_frequency: int = 20) -> None:
        """
        By default the data in a PyTorch Dataset is only generated when it is first accessed
        with __getitem__(). Since this occurs for most of the samples only when the training has started,
        it makes sense to cache the data in the dataset for speeding up the training.
        """
        for i in range(len(self.hashtags)):
            self[i] # by calling __getitem__() the results will be cached

            if i % report_frequency == 0:
                print(f'Cashing Dataset: {file_path} ... [{i} / {len(self.hashtags)}]')

        with open(file_path, 'wb') as file:
            pickle.dump(self._cache, file)

    @staticmethod
    def load_from_file(file_path: Path) -> 'CachedDataset':
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def __len__(self):
        return len(self.hashtags)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        hashtag = self.hashtags[idx]

        hashtag_emb = torch.tensor(self.word_emb_model.get_embedding(hashtag))
        tweets_containing_hashtag = self._get_tweets_by_hashtag(hashtag)
        sent_embs = [self.sent_emb_model._generate_embedding(tweet.text) for tweet in tweets_containing_hashtag]
        # have to wrap the mean in another tensor, otherwise the following exception occurs when training
        # "RuntimeError: Trying to backward through the graph a second time"
        avg_sent_emb = torch.tensor(torch.mean(torch.stack(sent_embs), dim=0))
        
        data = {'x': hashtag_emb, 'y': avg_sent_emb}

        self._cache[idx] = data

        return data

class DataModule(pl.LightningDataModule):
    def __init__(
        self, tweets: List[Tweet], word_emb_model: WordEmbeddingModel, 
        sent_emb_model: SentenceEmbeddingModel,
        batch_size: int = 2, train_val_test_split: Tuple[int] = (0.7, 0.1, 0.2)
    ):
        # super.__init__()
        self.prepare_data_per_node = False # TODO brauche ich das immer noch?

        self.tweets = tweets
        self.word_emb_model = word_emb_model
        self.sent_emb_model = sent_emb_model
        self.batch_size = batch_size

        unique_hashtags = self._get_unique_hashtags(tweets, word_emb_model)
        self.train_hashtags, self.val_hashtags, self.test_hashtags = \
            self._split_into_train_val_test(unique_hashtags, train_val_test_split)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @staticmethod
    def _get_unique_hashtags(tweets: List[Tweet], word_emb_model: WordEmbeddingModel) -> List[str]:
        unique_hashtags = list(set(hashtag for tweet in tweets for hashtag in tweet.hashtags))
        unique_hashtags = word_emb_model.remove_hashtags_not_part_of_the_vocab(unique_hashtags)
       
        return unique_hashtags

    @staticmethod
    def _split_into_train_val_test(hashtags: List[str], train_val_test_split: Tuple[int]) -> Tuple[List[str], List[str], List[str]]:
        random.shuffle(hashtags)

        num_train_hashtags = round(len(hashtags) * train_val_test_split[0])
        num_val_hashtags = round(len(hashtags) * train_val_test_split[1])

        train_hashtags = hashtags[:num_train_hashtags]
        val_hashtags = hashtags[num_train_hashtags:num_train_hashtags + num_val_hashtags]
        test_hashtags = hashtags[num_train_hashtags + num_val_hashtags:]

        return train_hashtags, val_hashtags, test_hashtags

    def prepare_data(self):
        pass
    
    def setup(self, stage: str = None, restore_from_file: bool = False):
        if restore_from_file:
            self.train_dataset = CachedDataset.load_from_file('save_files/train_dataset.pickle')
            self.val_dataset = CachedDataset.load_from_file('save_files/val_dataset.pickle')
            self.test_dataset = CachedDataset.load_from_file('save_files/test_dataset.pickle')
        else:
            self.train_dataset = CachedDataset(
                self.train_hashtags, self.tweets, self.word_emb_model, self.sent_emb_model
            )
            self.val_dataset = CachedDataset(
                self.val_hashtags, self.tweets, self.word_emb_model, self.sent_emb_model
            )
            self.test_dataset = CachedDataset(
                self.test_hashtags, self.tweets, self.word_emb_model, self.sent_emb_model
            )

            self.train_dataset.cache_all_and_store('save_files/train_dataset.pickle')
            self.val_dataset.cache_all_and_store('save_files/val_dataset.pickle')
            self.test_dataset.cache_all_and_store('save_files/test_dataset.pickle')

    @property
    def in_features(self) -> int:
        return self.train_dataset[0]['x'].shape[0]

    @property
    def out_features(self) -> int:
        return self.train_dataset[0]['y'].shape[0]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)