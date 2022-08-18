import matplotlib

import pickle

from hashtag_to_sent_mapper import Hashtag2SentMapper
from dataset import DataModule
from hashtag_analyzer import HashtagAnalyzer
from sentence_embedding_model import SentenceEmbeddingModel
from tweet_processor import TweetProcessor

if __name__ == '__main__':
    matplotlib.use('WebAgg')

    dataset_file_path = 'datasets/hashtags-en-tweets.jsonl'

    tweet_processor = TweetProcessor()

    sent_emb_tweets = tweet_processor.preprocess_tweets_for_sentence_embedding(dataset_file_path)

    word_emb_model = pickle.load(open('save_files/word_emb_model.pkl', 'rb'))
    sent_emb_model = SentenceEmbeddingModel()

    data_module = DataModule.restore_from_file()

    hashtag_to_sent_mapper = Hashtag2SentMapper.load_from_checkpoint(
        'save_files/best_model.ckpt',
        in_features=data_module.in_features,
        out_features=data_module.out_features,
        hidden_layer1_size=300, hidden_layer2_size=500
    )

    hashtag_analyzer = HashtagAnalyzer(sent_emb_tweets, word_emb_model, hashtag_to_sent_mapper)

    hashtag_analyzer.plot_hashtags_with_topics(
        'datasets/top_hashtags.csv', after_transformation=False,
        # selected_topics={'ukraine', 'society', 'sci_tech', 'middle_east', 'football'}
        # selected_topics={'ukraine', 'society', 'sci_tech', 'middle_east', 'football', 'tv_cinema', 'news', 'local_news', 'tv', 'politics'}
        use_merged_topics=True
    )

    
    hashtag_analyzer.plot_hashtags_with_topics(
        'datasets/top_hashtags.csv', after_transformation=True,
        # selected_topics={'ukraine', 'society', 'sci_tech', 'middle_east', 'football'}
        # selected_topics={'ukraine', 'society', 'sci_tech', 'middle_east', 'football', 'tv_cinema', 'news', 'local_news', 'tv', 'politics'}
        use_merged_topics=True
    )