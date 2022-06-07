import os
from pickle import dump, load

import constants as c
import dataset_handler as dh
import evaluation as ev
import topic_discovery as td
import model
from tweet_processor import TweetProcessor
from word_embedding_model import WordEmbeddingModel

def main():
    dataset_file_path = 'datasets/hashtags-en-tweets.jsonl'

    # creating folder for save files
    if not os.path.exists(c.SAVE_FOLDER):
        os.mkdir(c.SAVE_FOLDER)

    tweet_processor = TweetProcessor()
    print("> Preprocessing data for Word2Vec model")
    word_emb_tweets = tweet_processor.preprocess_tweets_for_word_embedding(dataset_file_path)

    print("> Training Word2Vec model")
    word_embedding_model = WordEmbeddingModel()
    word_embedding_model.train(word_emb_tweets)
    print("W2V MODEL TRAINING SUCCEDED")

    # Get data for Google Universal Sentence Encoder
    print("> Preprocessing data for sentence embeddings")
    sent_emb_tweets = tweet_processor.preprocess_tweets_for_sentence_embedding(dataset_file_path)

    train_tweets, test_tweets = dh.split_into_train_test(sent_emb_tweets)

    print("> Preparing data for neural network training")
    sentences_train, sentences_test, targets_train, targets_test, test_hashtags = dh.prepare_model_inputs_and_targets(
        train_tweets, test_tweets, word_embedding_model
    )

    print("> Training MLP model")
    model.transfer_and_fine_tune(sentences_train, sentences_test, targets_train, targets_test)
    print("> Loading MLP model and making predictions")
    model.predict_hashtags_and_store_results(word_embedding_model, test_tweets, sentences_test)

    # # Evaluate model
    # ht_lists = load(open(c.SAVE_FOLDER + 'ht_lists.pkl', 'rb'))

    # if c.EXPANSION_STRATEGY == c.LOCAL_EXPANSION:
    #     print("> Local expansion evaluation")
    #     ev.local_nhe_evaluation(ht_lists, sentences_test, model, c.MAX_EXPANSION_ITERATIONS)
    # else:
    #     print("> Global expansion evaluation")
    #     ev.global_nhe_evaluation(ht_lists, sentences_test, model, c.MAX_EXPANSION_ITERATIONS)

    # print("> Topic discovery evaluation")
    # td.evaluate_topic_discovery(model, c.TEST_CORPUS, c.MAX_EXPANSION_ITERATIONS)

if __name__ == '__main__':
    # main()
    dataset_file_path = 'datasets/hashtags-en-tweets.jsonl'
    tweet_processor = TweetProcessor()
    sent_emb_tweet_corpus = tweet_processor.preprocess_tweets_for_sentence_embedding(dataset_file_path)

    print('done')
