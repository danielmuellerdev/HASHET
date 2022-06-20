import json
from pathlib import Path
import random
import re
from pickle import dump
from pickle import load as pkload
from typing import List, Tuple

import numpy as np
import spacy
from spacy.tokens.doc import Doc as SpacyDoc
from spacy.tokens.token import Token as SpacyToken
import unidecode
import gensim

import constants as c
from word_embedding_model import WordEmbeddingModel
from tweet import Tweet

def split_into_train_test(
    tweets: List[str], testset_size_in_percent = c.TESTSET_SIZE_IN_PERCENT
) -> Tuple[List[Tweet], List[Tweet]]:

    random.shuffle(tweets)

    test_tweets = tweets[:round(len(tweets) * testset_size_in_percent)]
    train_tweets = tweets[len(test_tweets):]

    return train_tweets, test_tweets

def prepare_model_inputs_and_targets(
    train_tweets: List[Tweet], test_tweets: List[Tweet], word_emb_model: WordEmbeddingModel
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[str]]]:

    sentences_train, sentences_test = [], []
    targets_train, targets_test = [], []
    test_hashtags = []

    for tweets, is_test in [(train_tweets, False), (test_tweets, True)]:
        for tweet in tweets:
            hashtags = word_emb_model.remove_hashtags_not_part_of_the_vocab(tweet.hashtags)
            target_emb = word_emb_model.generate_target(hashtags)

            if target_emb is not None:
                if is_test:
                    sentences_test.append(tweet.text)
                    targets_test.append(target_emb)
                else:
                    sentences_train.append(tweet.text)
                    targets_train.append(target_emb)

                if is_test:
                    test_hashtags.append(hashtags)

    sentences_train, sentences_test = np.array(sentences_train), np.array(sentences_test)
    targets_train, targets_test = np.array(targets_train), np.array(targets_test)

    return sentences_train, sentences_test, targets_train, targets_test, test_hashtags
