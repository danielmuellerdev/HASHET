from typing import List, Tuple, Dict
from pathlib import Path
from collections import defaultdict
import random

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from torch import Tensor
from matplotlib.axes import Axes
from sentence_embedding_model import SentenceEmbeddingModel

from word_embedding_model import WordEmbeddingModel
from hashtag_to_sent_mapper import Hashtag2SentMapper
from tweet import Tweet
from dataset import CachedDataset


class HashtagVisualizer:
    COLORS = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]

    def __init__(
        self, sent_emb_tweets: List[Tweet], word_emb_model: WordEmbeddingModel, 
        hashtag_to_sent_mapper: Hashtag2SentMapper
    ):
        self.sent_emb_tweets = sent_emb_tweets
        self._word_emb_model = word_emb_model
        self._hashtag_to_sent_mapper = hashtag_to_sent_mapper
        
        self._pca = PCA(n_components=2)
        
        self.unique_hashtags = set(hashtag for tweet in sent_emb_tweets for hashtag in tweet.hashtags)
        self.unique_hashtags = word_emb_model.remove_hashtags_not_part_of_the_vocab(self.unique_hashtags)

    @staticmethod
    def _get_hashtag_embeddings(
        hashtags: List[str],
        word_emb_model: WordEmbeddingModel,
        hashtag_to_sent_mapper: Hashtag2SentMapper = None,
        after_transformation: bool = False
    ) -> Tensor:
        hashtag_embs = [word_emb_model.get_embedding(hashtag) for hashtag in hashtags]

        if after_transformation:
            with torch.no_grad():
                hashtag_embs = [hashtag_to_sent_mapper(hashtag_emb) for hashtag_emb in hashtag_embs]
        
        return torch.stack(hashtag_embs)

    @staticmethod
    def _reduce_to_two_dimensions(tensor: Tensor, pca: PCA) -> Tensor:
        return pca.fit_transform(tensor)

    def plot_all_hashtags(self, after_transformation: bool = False) -> None:
        hashtag_embs = self._get_hashtag_embeddings(
            self.unique_hashtags, self._word_emb_model, self._hashtag_to_sent_mapper, after_transformation
        )
        two_dim_hashtag_embs = self._reduce_to_two_dimensions(hashtag_embs, self._pca)

        x, y = two_dim_hashtag_embs[:, 0], two_dim_hashtag_embs[:, 1]

        plt.scatter(x, y)

        fig = plt.gcf()
        fig.set_size_inches(20, 10)
        
        plt.title(
            'All Hashtags in the Hashtag-embedding space '
            + ('[after transformation]' if after_transformation else '[Word2Vec]')
        )
        plt.savefig(
            'save_files/'
            + 'all_hashtags_' + ('transformed' if after_transformation else 'word2vec') + '.png'
        )
        plt.show()

    @staticmethod
    def _read_topics_file(
        topics_file_path: Path, word_emb_model: WordEmbeddingModel
    ) -> Tuple[List[str], List[str]]:
        hashtags, topics = [], []
        with open(topics_file_path, encoding='utf-8') as file:
            for line in file.readlines()[1:]: # skip header line
                hashtag, topic = line.strip().split(',')

                if hashtag not in word_emb_model.vocab:
                    continue

                hashtags.append(hashtag)
                topics.append(topic)
        
        return hashtags, topics

    @staticmethod
    def _label_selected_hashtags(
        hashtags: List[str], topics: List[str], ax: Axes, 
        num_hashtags_to_label_per_topic: int, x: Tensor, y: Tensor
    ) -> None:
        topic_to_num_hashtags_labeled = defaultdict(int)
        for i, (hashtag, topic) in enumerate(zip(hashtags, topics)):
            if topic_to_num_hashtags_labeled[topic] < num_hashtags_to_label_per_topic:
                ax.annotate(hashtag, (x[i], y[i]))

                topic_to_num_hashtags_labeled[topic] += 1

    def plot_hashtags_with_topics(
        self, topics_file_path: Path, after_transformation: bool = False,
        num_hashtags_to_label_per_topic: int = 3
     ) -> None:
        hashtags, topics = self._read_topics_file(topics_file_path, self._word_emb_model)

        hashtag_embs = self._get_hashtag_embeddings(
            hashtags, self._word_emb_model, self._hashtag_to_sent_mapper, after_transformation
        )
        two_dim_hashtag_embs = self._reduce_to_two_dimensions(hashtag_embs, self._pca)

        x, y = two_dim_hashtag_embs[:, 0], two_dim_hashtag_embs[:, 1]
        
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)
        
        topic_to_color_num = {topic: i for i, topic in enumerate(set(topics))}
        topic_to_stats = defaultdict(list)
        for i, topic in enumerate(topics):
            topic_to_stats[topic].append((x[i], y[i]))

        for topic, stats in topic_to_stats.items():
            xs, ys = [x for x, _ in stats], [y for _, y in stats]
            color = HashtagVisualizer.COLORS[topic_to_color_num[topic]]
            ax.scatter(xs, ys, c=color, label=topic)

        ax.legend()

        self._label_selected_hashtags(hashtags, topics, ax, num_hashtags_to_label_per_topic, x, y)
        
        plt.title(
            'Selected Hashtags with their topics in the Hashtag-embedding space '
            + ('[after transformation]' if after_transformation else '[Word2Vec]')
        )
        plt.savefig(
            'save_files/'
            + 'selected_hashtags_with_topics_' + ('transformed' if after_transformation else 'word2vec') + '.png'
        )
        plt.show()

    def plot_a_hashtag(self, hashtag: str, sent_emb_model: SentenceEmbeddingModel) -> None:
        hashtag_emb = self._get_hashtag_embeddings(
            [hashtag], self._word_emb_model, self._hashtag_to_sent_mapper
        )[0]

        transformed_hashtag_emb = self._get_hashtag_embeddings(
            [hashtag], self._word_emb_model, self._hashtag_to_sent_mapper, after_transformation=True
        )[0]
        
        hashtag_to_tweets = defaultdict(list) # TODO: als TweetManager Klasse?
        for tweet in self.sent_emb_tweets:
            for tweet_hashtag in tweet.hashtags:
                hashtag_to_tweets[tweet_hashtag].append(tweet)

        tweets_containing_hashtag = hashtag_to_tweets[hashtag] 

        sent_embs = [sent_emb_model.generate_embedding(tweet.text) for tweet in tweets_containing_hashtag]

        avg_sent_emb = CachedDataset._memory_efficient_mean(
            tweets_containing_hashtag,
            lambda tweet: sent_emb_model.generate_embedding(tweet.text),
            dim=sent_emb_model.OUTPUT_DIM
        )

        stacked_embs = torch.stack([avg_sent_emb] + [transformed_hashtag_emb] + sent_embs)
        two_dim_embs = self._reduce_to_two_dimensions(stacked_embs, self._pca)

        x, y = two_dim_embs[:, 0], two_dim_embs[:, 1]

        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)

        ax.scatter(x[0], y[0], c=HashtagVisualizer.COLORS[1], label='centroid')
        ax.scatter(x[1], y[1], c=HashtagVisualizer.COLORS[3], label='predicted centroid')
        ax.scatter(x[2:], y[2:], c=HashtagVisualizer.COLORS[7], label='sentence embeddings')
        
        ax.legend()
        plt.title(
            f'Transformed hashtag-embedding for hashtag: "{hashtag}" in the '
            f'sentence embedding space of all the tweets containing the hashtag ({len(tweets_containing_hashtag)})'
        )

        plt.show()

