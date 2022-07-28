from collections import defaultdict
from typing import List, Tuple, Iterable, Callable
from pathlib import Path
import pickle
import random
import time
import datetime

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch import Tensor

from tweet import Tweet
from word_embedding_model import WordEmbeddingModel
from sentence_embedding_model import SentenceEmbeddingModel


class CachedDataset(torch.utils.data.Dataset):
    def __init__(
        self, hashtags: List[str], tweets: List[Tweet], word_emb_model: WordEmbeddingModel,
        sent_emb_model: SentenceEmbeddingModel, num_hashtags_per_sent_emb_limit: int = None
    ):
        self.hashtags = hashtags
        self.word_emb_model = word_emb_model
        self.sent_emb_model = sent_emb_model
        self.num_hashtags_per_sent_emb_limit = num_hashtags_per_sent_emb_limit

        self._hashtag_to_tweets = defaultdict(list) # TODO: als TweetManager Klasse?
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
        print(f'Cashing Dataset: {file_path} ...')
        start = time.time()
        for i in range(len(self.hashtags)):
            self[i] # by calling __getitem__() the results will be cached

            if i > 0 and i % report_frequency == 0:
                time_run = time.time() - start
                percent_done = i / len(self.hashtags) * 100
                percent_left = 100 - percent_done
                remaining_time_in_s = time_run / percent_done * percent_left
                h, m, s = str(datetime.timedelta(seconds=remaining_time_in_s)).split(':')
                remaining_time_str = f'{h}h:{m}m:{int(float(s)):02d}s'
                estimated_finish_time = datetime.datetime.now() + datetime.timedelta(seconds=remaining_time_in_s)

                print(
                    f'Cashing Dataset: {file_path} ... [{i:4d} / {len(self.hashtags):4d}]'
                    f' [Remaining: {remaining_time_str}] [ETA: {estimated_finish_time}]'
                )

        with open(file_path, 'wb') as file:
            pickle.dump(self._cache, file)
        
        print(f'Cashing Dataset: {file_path} finished')

    @staticmethod
    def load_from_file(file_path: Path) -> 'CachedDataset':
        with open(file_path, 'rb') as file:
            cache = pickle.load(file)
        
        cached_dataset = CachedDataset([], [], None, None, None)
        cached_dataset._cache = cache

        return cached_dataset

    def __len__(self):
        return len(self.hashtags) if len(self._cache) == 0 else len(self._cache)

    @staticmethod
    def _memory_efficient_mean(items: Iterable, func: Callable, dim: int) -> Tensor:
        """
        Won't overflow and shouldn't underflow. Applies the given function to each item one by one
        and only holds one item at a time in memory after applying the function.
        """
        result = torch.zeros(dim)
        for item in items:
            result += (1 / len(items)) * func(item)

        return result

    def _generate_hashtag_embedding(self, hashtag: str) -> Tensor:
        return self.word_emb_model.get_embedding(hashtag)

    def _generate_averaged_sentence_embedding(self, hashtag: str) -> Tensor:
        tweets_containing_hashtag = self._get_tweets_by_hashtag(hashtag)
        
        if self.num_hashtags_per_sent_emb_limit is not None:
            random.shuffle(tweets_containing_hashtag) # TODO random.sample()
            tweets_containing_hashtag = tweets_containing_hashtag[:self.num_hashtags_per_sent_emb_limit]
        
        avg_sent_emb = self._memory_efficient_mean(
            tweets_containing_hashtag,
            lambda tweet: self.sent_emb_model.generate_embedding(tweet.text),
            dim=self.sent_emb_model.OUTPUT_DIM
        )

        return avg_sent_emb

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        hashtag = self.hashtags[idx]

        data = {
            'x': self._generate_hashtag_embedding(hashtag),
            'y': self._generate_averaged_sentence_embedding(hashtag)
        }

        self._cache[idx] = data

        return data


class DataModule(pl.LightningDataModule):
    def __init__(
        self, tweets: List[Tweet], word_emb_model: WordEmbeddingModel,
        sent_emb_model: SentenceEmbeddingModel,
        batch_size: int = 1, train_val_test_split: Tuple[int] = (0.7, 0.1, 0.2),
        num_hashtags_per_sent_emb_limit: int = None,
        num_workers: int = 1
    ):
        super().__init__()
        self.prepare_data_per_node = False # TODO brauche ich das immer noch?

        self._tweets = tweets
        self._word_emb_model = word_emb_model
        self._sent_emb_model = sent_emb_model
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_hashtags_per_sent_emb_limit = num_hashtags_per_sent_emb_limit
        self.num_workers = num_workers

        self.is_restored_from_file = False
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.split = None

    @staticmethod
    def restore_from_file(split: int = None) -> 'DataModule':
        if split is not None:
            train_file_path = f'save_files/train_dataset_split_{split}.pkl'
            val_file_path = f'save_files/val_dataset_split_{split}.pkl'
            test_file_path = f'save_files/test_dataset_split_{split}.pkl'
        else:
            train_file_path = 'save_files/train_dataset.pkl'
            val_file_path = 'save_files/val_dataset.pkl'
            test_file_path = 'save_files/test_dataset.pkl'

        train_dataset = CachedDataset.load_from_file(train_file_path)
        val_dataset = CachedDataset.load_from_file(val_file_path)
        test_dataset = CachedDataset.load_from_file(test_file_path)

        data_module = DataModule(None, None, None)
        data_module.train_dataset = train_dataset
        data_module.val_dataset = val_dataset
        data_module.test_dataset = test_dataset
        data_module.is_restored_from_file = True
        data_module.split = split

        return data_module

    @staticmethod
    def _get_unique_hashtags(tweets: List[Tweet], word_emb_model: WordEmbeddingModel) -> List[str]:
        unique_hashtags = list(set(hashtag for tweet in tweets for hashtag in tweet.hashtags))
        unique_hashtags_in_w2v_vocab = word_emb_model.remove_hashtags_not_part_of_the_vocab(unique_hashtags)
       
        return unique_hashtags_in_w2v_vocab

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
    
    def setup(self, stage: str = None):
        if self.is_restored_from_file:
            return

        unique_hashtags = self._get_unique_hashtags(self._tweets, self._word_emb_model)
        train_hashtags, val_hashtags, test_hashtags = \
            self._split_into_train_val_test(unique_hashtags, self.train_val_test_split)

        self.train_dataset = CachedDataset(
            train_hashtags, self._tweets, self._word_emb_model, self._sent_emb_model,
            self.num_hashtags_per_sent_emb_limit
        )
        self.val_dataset = CachedDataset(
            val_hashtags, self._tweets, self._word_emb_model, self._sent_emb_model,
            self.num_hashtags_per_sent_emb_limit
        )
        self.test_dataset = CachedDataset(
            test_hashtags, self._tweets, self._word_emb_model, self._sent_emb_model,
            self.num_hashtags_per_sent_emb_limit
        )

        self.train_dataset.cache_all_and_store('save_files/train_dataset.pkl')
        self.val_dataset.cache_all_and_store('save_files/val_dataset.pkl')
        self.test_dataset.cache_all_and_store('save_files/test_dataset.pkl')

    @property
    def in_features(self) -> int:
        return self.train_dataset[0]['x'].shape[0]

    @property
    def out_features(self) -> int:
        return self.train_dataset[0]['y'].shape[0]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)