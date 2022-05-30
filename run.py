import os
from pickle import dump, load

import constants as c
import dataset_handler as dh
import evaluation as ev
import topic_discovery as td
import model
from word_embedding_model import WordEmbeddingModel

def main():
    dataset_file_paths = ['datasets/hashtags-en-tweets.jsonl']
    is_custom_dataset = True

    # creating folder for save files
    if not os.path.exists(c.SAVE_FOLDER):
        os.mkdir(c.SAVE_FOLDER)

    print("> Preprocessing data for Word2Vec model")
    tweet_corpus = dh.preprocess_tweets(dataset_file_paths, save_file=c.W2V_INPUT, is_custom_dataset=is_custom_dataset)
    
    print("> Training Word2Vec model")
    word_embedding_model = WordEmbeddingModel()
    word_embedding_model.train(tweet_corpus)
    print("W2V MODEL TRAINING SUCCEDED")

    # Get data for Google Universal Sentence Encoder
    print("> Preprocessing data for sentence embeddings")
    dh.preprocess_data_for_sentence_embedding(dataset_file_paths, is_custom_dataset=True)

    dh.prepare_train_test()

    print("> Preparing data for neural network training")
    tweet_corpus_train, tweet_corpus_test, targets_train, targets_test, hashtags = dh.prepare_model_inputs_and_targets(
        word_embedding_model)


    # dump(sentences_train, open(c.SAVE_FOLDER + 'sentences_train.pkl', 'wb'))
    # dump(sentences_test, open(c.SAVE_FOLDER + 'sentences_test.pkl', 'wb'))
    # dump(targets_train, open(c.SAVE_FOLDER + 'targets_train.pkl', 'wb'))
    # dump(targets_test, open(c.SAVE_FOLDER + 'targets_test.pkl', 'wb'))
    # dump(ht_lists, open(c.SAVE_FOLDER + 'ht_lists.pkl', 'wb'))

    # sentences_train = load(open(c.SAVE_FOLDER + 'sentences_train.pkl', 'rb'))
    # sentences_test = load(open(c.SAVE_FOLDER + 'sentences_test.pkl', 'rb'))
    # targets_train = load(open(c.SAVE_FOLDER + 'targets_train.pkl', 'rb'))
    # targets_test = load(open(c.SAVE_FOLDER + 'targets_test.pkl', 'rb'))

    # print("> Training MLP model")
    # model.transfer_and_fine_tune(sentences_train, sentences_test, targets_train, targets_test)
    # print("> Loading MLP model and making predictions")
    # model.predict_hashtags_and_store_results(c.TEST_CORPUS, sentences_test)

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
    main()
