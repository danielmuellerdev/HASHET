from typing import Dict, List, Set, Tuple
from pathlib import Path
from collections import defaultdict
import random

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch import Tensor
from matplotlib.axes import Axes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

from sentence_embedding_model import SentenceEmbeddingModel
from word_embedding_model import WordEmbeddingModel
from hashtag_to_sent_mapper import Hashtag2SentMapper
from tweet import Tweet
from dataset import CachedDataset, DataModule


class HashtagAnalyzer:
    COLORS = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
        'b', 'gold', 'lime', 'navy'
    ]

    def __init__(
        self, sent_emb_tweets: List[Tweet], word_emb_model: WordEmbeddingModel,
        hashtag_to_sent_mapper: Hashtag2SentMapper,
        pca_explained_variance_treshhold: float = 0.95
    ):
        self.sent_emb_tweets = sent_emb_tweets
        self._word_emb_model = word_emb_model
        self._hashtag_to_sent_mapper = hashtag_to_sent_mapper
        
        self._pca = PCA(n_components=pca_explained_variance_treshhold, svd_solver='full')
        
        self.unique_hashtags = set(
            hashtag
            for tweet in sent_emb_tweets
            for hashtag in tweet.hashtags
            if hashtag in word_emb_model.vocab
        )

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
    def _reduce_to_n_dimensions(tensor: Tensor, pca: PCA, n: int) -> Tensor:
        # reduce as many dims as possible while keeping the explained variance above a given treshhold
        tensor_with_reduced_dims = pca.fit_transform(tensor)

        tsne = TSNE(n_components=n)

        return tsne.fit_transform(tensor_with_reduced_dims)

    def plot_all_hashtags(self, after_transformation: bool = False, save_to_file: bool = False) -> None:
        hashtag_embs = self._get_hashtag_embeddings(
            self.unique_hashtags, self._word_emb_model, self._hashtag_to_sent_mapper, after_transformation
        )
        two_dim_hashtag_embs = self._reduce_to_n_dimensions(hashtag_embs, self._pca, n=2)

        x, y = two_dim_hashtag_embs[:, 0], two_dim_hashtag_embs[:, 1]

        plt.scatter(x, y)

        fig = plt.gcf()
        fig.set_size_inches(20, 10)
        
        plt.title(
            'All Hashtags in the Hashtag-embedding space '
            + ('[after transformation]' if after_transformation else '[Word2Vec]')
        )
        
        if save_to_file:
            plt.savefig(
                'save_files/'
                + 'all_hashtags_' + ('transformed' if after_transformation else 'word2vec') + '.png'
            )
        
        plt.show()

    @staticmethod
    def _read_topics_file(
        topics_file_path: Path, word_emb_model: WordEmbeddingModel,
        use_merged_topics: bool, selected_topics: Set[str] = None, seperator: str = ';'
    ) -> Tuple[List[str], List[str]]:
        hashtags, topics = [], []
        with open(topics_file_path, encoding='utf-8') as file:
            for line in file.readlines():
                hashtag, topic, merged_topic = line.strip().split(seperator)[:3]

                if use_merged_topics:
                    topic = merged_topic

                if (hashtag not in word_emb_model.vocab) or \
                    (selected_topics is not None and topic not in selected_topics):
                    continue

                hashtags.append(hashtag)
                topics.append(topic)
        
        return hashtags, topics

    @staticmethod
    def _label_selected_hashtags(
        hashtags: List[str], topics: List[str], ax: Axes,
        num_hashtags_to_label_per_topic: int, x: Tensor, y: Tensor, z: Tensor
    ) -> None:
        topic_to_num_hashtags_labeled = defaultdict(int)
        for i, (hashtag, topic) in enumerate(zip(hashtags, topics)):
            if topic_to_num_hashtags_labeled[topic] < num_hashtags_to_label_per_topic:
                ax.text(x[i], y[i], z[i], hashtag)

                topic_to_num_hashtags_labeled[topic] += 1

    def plot_hashtags_with_topics(
        self, topics_file_path: Path, after_transformation: bool = False,
        num_hashtags_to_label_per_topic: int = 2, save_to_file: bool = False,
        selected_topics: Set[str] = None, use_merged_topics: bool = False
     ) -> None:
        """ Plots in 3D. """

        hashtags, topics = self._read_topics_file(
            topics_file_path, self._word_emb_model, use_merged_topics, selected_topics=selected_topics
        )

        hashtag_embs = self._get_hashtag_embeddings(
            hashtags, self._word_emb_model, self._hashtag_to_sent_mapper, after_transformation
        )
        three_dim_hashtag_embs = self._reduce_to_n_dimensions(hashtag_embs, self._pca, n=3)

        x, y, z = three_dim_hashtag_embs[:, 0], three_dim_hashtag_embs[:, 1], three_dim_hashtag_embs[:, 2]

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig.set_size_inches(20, 10)
        
        unique_topics_ordered = list(dict.fromkeys(topics))
        topic_to_color_num = {topic: i for i, topic in enumerate(unique_topics_ordered)}
        topic_to_stats = defaultdict(list)
        for i, topic in enumerate(topics):
            topic_to_stats[topic].append((x[i], y[i], z[i]))

        for topic, stats in topic_to_stats.items():
            xs, ys, zs = [x for x, _, _ in stats], [y for _, y, _ in stats], [z for _, _, z in stats]
            color = HashtagAnalyzer.COLORS[topic_to_color_num[topic]]
            ax.scatter(xs, ys, zs, c=color, label=topic)

        ax.legend()

        self._label_selected_hashtags(hashtags, topics, ax, num_hashtags_to_label_per_topic, x, y, z)
        
        plt.title(
            'Selected Hashtags with their topics in the Hashtag-embedding space '
            + ('[after transformation]' if after_transformation else '[Word2Vec]')
        )
        
        if save_to_file:
            plt.savefig(
                'save_files/'
                + 'selected_hashtags_with_topics_' + ('transformed' if after_transformation else 'word2vec') + '.png'
            )
        
        plt.show()

    def plot_a_hashtag(self, hashtag: str, sent_emb_model: SentenceEmbeddingModel) -> None:
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
        two_dim_embs = self._reduce_to_n_dimensions(stacked_embs, self._pca, n=2)

        x, y = two_dim_embs[:, 0], two_dim_embs[:, 1]

        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)

        ax.scatter(x[0], y[0], c=HashtagAnalyzer.COLORS[1], label='centroid')
        ax.scatter(x[1], y[1], c=HashtagAnalyzer.COLORS[3], label='predicted centroid')
        ax.scatter(x[2:], y[2:], c=HashtagAnalyzer.COLORS[7], label='sentence embeddings')
        
        ax.legend()
        plt.title(
            f'Transformed hashtag-embedding for hashtag: "{hashtag}" in the '
            f'sentence embedding space of all the tweets containing the hashtag ({len(tweets_containing_hashtag)})'
        )

        plt.show()

    def plot_sentence_embeddings_vs_centroids(
        self, data_module: DataModule, sent_emb_model: SentenceEmbeddingModel, num_sent_embs_to_plot: int = 500, report_freq: int = 20
    ) -> None:
        tweets_to_plot = random.sample(self.sent_emb_tweets, num_sent_embs_to_plot)

        sent_embs_to_plot = []
        for i, tweet in enumerate(tweets_to_plot):
            sent_embs_to_plot.append(sent_emb_model.generate_embedding(tweet.text))

            if i % report_freq == 0:
                print(f'retrieving sentence embeddings ... [{i} / {num_sent_embs_to_plot}]')

        data_module.batch_size = 1

        train_centroids = [batch['y'][0] for batch in data_module.train_dataloader()]
        val_centroids = [batch['y'][0] for batch in data_module.val_dataloader()]
        test_centroids = [batch['y'][0] for batch in data_module.test_dataloader()]

        stacked_centroids = torch.stack(train_centroids + val_centroids + test_centroids + sent_embs_to_plot)

        two_dim_stacked_centroids = self._reduce_to_n_dimensions(stacked_centroids, self._pca, n=2)

        two_dim_train_centroids = two_dim_stacked_centroids[:len(train_centroids)]
        two_dim_val_centroids = two_dim_stacked_centroids[len(train_centroids):len(train_centroids) + len(val_centroids)]
        two_dim_test_centroids = two_dim_stacked_centroids[len(train_centroids) + len(val_centroids):len(train_centroids) + len(val_centroids) + len(test_centroids)]
        two_dim_sent_embs = two_dim_stacked_centroids[len(train_centroids) + len(val_centroids) + len(test_centroids):]

        train_x, train_y = two_dim_train_centroids[:, 0], two_dim_train_centroids[:, 1]
        val_x, val_y = two_dim_val_centroids[:, 0], two_dim_val_centroids[:, 1]
        test_x, test_y = two_dim_test_centroids[:, 0], two_dim_test_centroids[:, 1]
        sent_embs_x, sent_embs_y = two_dim_sent_embs[:, 0], two_dim_sent_embs[:, 1]

        print(stacked_centroids.shape, len(two_dim_train_centroids), len(two_dim_val_centroids), len(two_dim_test_centroids), len(two_dim_sent_embs))

        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)

        ax.scatter(train_x, train_y, c=HashtagAnalyzer.COLORS[0], label='train centroids')
        ax.scatter(val_x, val_y, c=HashtagAnalyzer.COLORS[1], label='validation centroids')
        ax.scatter(test_x, test_y, c=HashtagAnalyzer.COLORS[2], label='test centroids')
        ax.scatter(sent_embs_x, sent_embs_y, c=HashtagAnalyzer.COLORS[3], label='sentence embeddings (random sampled)')

        ax.legend()

        plt.show()

    def calculate_hashtag_embedding_metrics(
        self, topics_file_path: Path, after_transformation: bool = False,
        num_retries: int = 20, use_merged_topics: bool = False
    ) -> Dict[str, float]:
        hashtags, topics = self._read_topics_file(topics_file_path, self._word_emb_model, use_merged_topics)

        encoded_topics = LabelEncoder().fit_transform(topics)

        hashtag_embs = self._get_hashtag_embeddings(
            hashtags, self._word_emb_model, self._hashtag_to_sent_mapper, after_transformation
        )

        num_topics = len(set(topics))

        silhouette_scores, ami_scores = [], []
        for _ in range(num_retries):
            clustered_hashtag_embs = KMeans(n_clusters=num_topics).fit_predict(hashtag_embs)

            silhouette_scores.append(
                silhouette_score(hashtag_embs, clustered_hashtag_embs)
            )
            ami_scores.append(
                adjusted_mutual_info_score(clustered_hashtag_embs, encoded_topics)
            )

        return {
                'clusters': num_topics,
                'mean_silhouette': np.mean(silhouette_scores),
                'mean_AMI': np.mean(ami_scores)
            }
