#!/usr/bin/env python
import json
from pathlib import Path
import random
import numpy as np
import re  # For preprocessing
from pickle import dump
from pickle import load as pkload
from typing import List

import spacy
from spacy.tokens.doc import Doc as SpacyDoc
from spacy.tokens.token import Token as SpacyToken
import unidecode

import emoji
import gensim

import constants as c
import word_embedding_model as emb

URL_REGEX = r"@\w*|https?:?\/?\/?[\w.\/]*|https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"
USER_NAMES_REGEX = re.compile(r'@\S+')
ACRONYM_REGEX = re.compile(r'[A-Z]{2,}$')
HASHTAG_REGEX = re.compile(r'#\S+')
LETTERS_AND_UNDERSCORES_REGEX = re.compile(r'[^\w+_ ]+')
WORDS_TO_REMOVE = ['rt', 'ht', 'htt', 'https', 'http', 'https t']


def _cleaning(doc: SpacyDoc) -> str:
    # tokens = capture_bigrams(words_doc) # ? ist der Output tokens? warhscheiunliuch eher List str
    tokens = list(doc)
    tokens = _remove_stopwords(tokens)
    tokens = _lemmatize(tokens)
    return ' '.join(tokens)

def _remove_stopwords(tokens: List[SpacyToken]) -> List[SpacyToken]:
    return [token for token in tokens if not token.is_stop and token.text not in WORDS_TO_REMOVE] # TODO test

def _lemmatize(
    tokens: List[SpacyToken],
    ignore_acronyms: bool = True,
    ignore_hashtags: bool = True
    ) -> List[SpacyToken]:
    
    lemmatized_tokens = []
    is_hashtag = False
    for token in tokens:
        if token.text == "#":
            is_hashtag = True
        elif ignore_hashtags and is_hashtag:
            lemmatized_tokens.append('#' + token.text)
            is_hashtag = False
        elif ignore_acronyms and _is_acronym(token.text):
            lemmatized_tokens.append(token.text)
        else:
            lemmatized_tokens.append(token.lemma_)

    return lemmatized_tokens
    

# def _cleaning(words_doc: SpacyDoc) -> str:
#     """Remove Stopwords and lemmatize all words except hashtags."""
#     txt = []
#     is_hashtag = False
#     for token in words_doc:
#         if token.text == "#":
#             is_hashtag = True
#         elif not (token.is_stop or (len(token.text) < 2 and token.text not in not_lemmatize)):
#             if is_hashtag:
#                 txt.append('#' + token.text)
#                 is_hashtag = False
#             else:
#                 txt.append(token.text if token.text in not_lemmatize else token.lemma_)
#     return ' '.join(txt)


def _get_tweet_corpus(file_paths: List[str], filter_out_retweets: bool = True, is_custom_dataset: bool = False) -> List[List[str]]:
    print('Read files')
    tweet_corpus = []
    for file_path in file_paths:
        with open(file_path, encoding='utf-8') as file:
            tweets = [json.loads(line.strip()) for line in file]

        for tweet in tweets:
            if is_custom_dataset:
                is_retweet = tweet['quoted_tweet_id'] is not None
            else:
                is_retweet = tweet['isRetweet']
                
            # skip tweet without text and retweets (no benefits for the embedding phase)
            if len(tweet['text']) == 0 or (filter_out_retweets and is_retweet):
                continue

            words = tweet['text'].split()
            tweet_corpus.append(words)

    # test
    # tweet_corpus = ["Hello this is a test of user's manual construction site voter San Francisco voted voter vote Cloud computing, major manufacturing companies. :) #matters 🙈".split()]

    return tweet_corpus

def _is_acronym(text: str) -> str:
    return ACRONYM_REGEX.match(text) is not None

def _remove_user_names(text: str) -> str:
    return re.sub(USER_NAMES_REGEX, '', text)

def _remove_urls(text: str) -> str:
    return re.sub(URL_REGEX, '', text).strip()

def _handle_accented_characters(accented_string: str) -> str:
    """Replace accented characters with unaccented ones: 'ü' -> 'ue'. """
    unaccented_string = unidecode.unidecode(accented_string)
    return unaccented_string

def _to_lowercase(text: str) -> str:
    """Lowercases the entire input text except acronyms. """
    lowercased_tokens = []
    for token in text.split():
        lowercase_token = token.lower() if not _is_acronym(token) else token
        lowercased_tokens.append(lowercase_token)
    
    return ' '.join(lowercased_tokens)

def _remove_hashtags(text: str) -> str:
    return re.sub(HASHTAG_REGEX, '', text)

def _remove_all_except_letters_and_underscores(text: str) -> str:
    return re.sub(LETTERS_AND_UNDERSCORES_REGEX, '', text).strip()

def _clean_and_phrase(tweet_corpus: List[List[str]], spacy_batch_size: int = 100) -> List[str]:
    cleaned_corpus = []
    for tweet_tokens in tweet_corpus:
        tweet = ' '.join(tweet_tokens)
        tweet = _remove_urls(tweet)
        tweet = _remove_hashtags(tweet)
        tweet = _remove_user_names(tweet)
        tweet = _handle_accented_characters(tweet)
        tweet = _to_lowercase(tweet)
        tweet = _remove_all_except_letters_and_underscores(tweet)

        cleaned_corpus.append(tweet.split())

    cleaned_corpus_with_bigrams = []
    bigrams = gensim.models.Phrases(cleaned_corpus)
    for tweet_tokens in cleaned_corpus:
        tweet_tokens_with_bigrams = bigrams[tweet_tokens] # new york -> new_york TODO nur nounphrases zulassen? dafür doch spacy nehmen?
        cleaned_corpus_with_bigrams.append(' '.join(tweet_tokens_with_bigrams))

    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

    # Taking advantage of spaCy .pipe() attribute to speed-up the cleaning process:
    txt = []
    for i, doc in enumerate(nlp.pipe(cleaned_corpus_with_bigrams, batch_size=spacy_batch_size, n_process=4)):
        cleaned_tweet = _cleaning(doc)

        # remove all characters except: words, digits and hashtag-symbols
        # cleaned_tweet = re.sub('[^#\\d\\w_]+', ' ', cleaned_tweet).strip()
        txt.append(cleaned_tweet)
        if i % (spacy_batch_size * 20) == 0:
            print(f'nlp.pipe: [{i} / {len(cleaned_corpus)}]')

    return txt


def _store(corpus, file):
    """
        Store list of list sentences in text file

        :param1 file.
        :return: corpus represented in list of list of words
    """
    with open(file, 'w', encoding="utf8") as stream_out:
        for line in corpus:
            for word in line:
                stream_out.write(str(word) + ' ')
            stream_out.write('\n')


def load(file):
    """
        Load text file in list of list sentences

        :param1 file.
        :return: corpus represented in list of list of words
    """
    sentences = []
    with open(file, "r", encoding="utf8") as stream_in:
        for line in stream_in:
            sentences.append(line.split())
    return sentences


def preprocess_data(file_paths: List[str], save_file: Path = 'sentences.txt', is_custom_dataset: bool = False):
    """
        Read data, clean data, calculate bigrams and store in save_file for Word2Vec model

        :param1 save_file.
    """
    tweet_corpus = _get_tweet_corpus(file_paths, filter_out_retweets=False, is_custom_dataset=is_custom_dataset)
    sentences = _clean_and_phrase(tweet_corpus)
    with open(save_file, 'w', encoding="utf8") as stream_out:
        for line in sentences:
            stream_out.write(line+'\n')


def preprocess_data_for_sentence_embedding(dataset_file_paths: List[Path], is_custom_dataset: bool = False) -> None:
    """
        Clean sentences and stores them in file for Google Universal Sentence Encoder

        :param1 file: specify input file in order to not use the whole input folder.
    """
    tweet_corpus = _get_tweet_corpus(dataset_file_paths, is_custom_dataset=is_custom_dataset)
    corpus_cleaned = [] # TODO hier müsste eigentlich das gleiche gemnacht werden, wie bei preprocessdata
    for tweet in tweet_corpus:
        tweet_cleaned = []
        tweet = re.sub(URL_REGEX, '', ' '.join(tweet)).strip()
        tweet = re.sub(r'[^\x00-\x7f]', r' ', tweet).lower().strip()
        tweet = tweet.split()
        for word in tweet:
            word_cleaned = word
            # word_cleaned = re.sub("[^#*\d*\w+_*]+", ' ', url_removal).strip()

            for r in WORDS_TO_REMOVE:
                if r == word_cleaned:
                    word_cleaned = ''
            if word_cleaned != '':
                tweet_cleaned.append(word_cleaned)

        # print(' '.join(tweet_cleaned))
        corpus_cleaned.append(tweet_cleaned)
    _store(corpus_cleaned, c.TRAIN_TEST_INPUT)


# ---------------------------------------------------------------------------------------------------------
def load_without_not_relevant_hts(input: Path, min_count: int = c.MINCOUNT) -> List[str]:
    counter = dict()
    with open(input, 'r', encoding="utf8") as in_stream:
        for line in in_stream:
            l = line.split()
            for w in l:
                if '#' in w:
                    counter[w] = counter.get(w, 0) + 1
   
    result = []
    with open(input, 'r', encoding="utf8") as in_stream:
        for line in in_stream:
            l = line.split()
            res_line = []
            for w in l:
                if '#' in w:
                    if counter[w] > min_count:
                        res_line.append(w)
                else:
                    res_line.append(w)
            if len(res_line) > 0:
                result.append(res_line)
    return result


def prepare_train_test(
        perc_test, 
        train_test_input: Path = c.TRAIN_TEST_INPUT,
        train_corpus: Path = c.TRAIN_CORPUS,
        test_corpus: Path = c.TEST_CORPUS
    ):
    """
        Split corpus in train and test and store them

        :param1 perc_test: test corpus percentage in float.
    """
    corpus_with_bigrams = load_without_not_relevant_hts(train_test_input)  # emb.load(TRAIN_TEST_INPUT) #
    random.shuffle(corpus_with_bigrams)
    cleaned_corpus_with_bigrams = []
    # keeping tweets with at least one hashtag and one word
    for tweet in corpus_with_bigrams:
        bool_h = False
        bool_w = False
        for w in tweet:
            if w == 'rt':
                break  # each retweet starts with 'rt'
            if '#' in w:
                bool_h = True
            else:
                bool_w = True
        if bool_w and bool_h:
            cleaned_corpus_with_bigrams.append(tweet)
    _store(cleaned_corpus_with_bigrams[int(len(cleaned_corpus_with_bigrams) * perc_test):], train_corpus)
    _store(cleaned_corpus_with_bigrams[:int(len(cleaned_corpus_with_bigrams) * perc_test)], test_corpus)


def hashtags_list(tweet, model):
    ht_list = []
    for w in tweet:
        word_cleaned = re.sub('[^#\\d\\w_]+', ' ', w).lower().strip()
        for word in word_cleaned.split():
            if word[0] == '#':
                word_cleaned = word
                break
        for r in emb.REMOVE_WORDS:
            if r == word_cleaned:
                word_cleaned = ''
        if len(word_cleaned) > 1 and '#' in word_cleaned and word_cleaned in model.wv.vocab: 
            ht_list.append(word_cleaned)
    return ht_list


def _count_words(tweets):
    counter = dict()
    for line in tweets:
        l = line.split()
        for w in l:
            counter[w] = counter.get(w, 0) + 1
    return counter


def _remove_hashtags_from_sentences(
        tweets,
        hts,
        populate_dictionary=True, 
        skip_hashtag_removal: bool = c.SKIP_HASHTAG_REMOVAL,
        h_removing_dict: Path = c.H_REMOVING_DICT
    ):
    if skip_hashtag_removal:
        result_tweets = []
        for tweet in tweets:
            tweet_string = ""
            for w in tweet:
                if w[0] == '#':
                    tweet_string = tweet_string + " " + w[1:]
                else:
                    tweet_string = tweet_string + " " + w
            result_tweets.append(tweet_string.strip())
        return result_tweets, hts

    if populate_dictionary:
        counter = _count_words(tweets)
        dump(counter, open(h_removing_dict, 'wb'))
    else:
        counter = pkload(open(h_removing_dict, 'rb'))
    result_tweets = []
    result_hts = []
    for tweet, ht_list in zip(tweets, hts):
        norm_tweet = []
        tweet = tweet.split()
        for word in tweet:
            if word[0] == '#':
                no_ht_word = word[1:]
                if counter.__contains__(no_ht_word) and counter[no_ht_word] > 2:  # mincount
                    norm_tweet.append(no_ht_word)
            else:
                norm_tweet.append(word)
        if len(norm_tweet) > 0:
            result_tweets.append(" ".join(norm_tweet))
            result_hts.append(ht_list)
    return result_tweets, result_hts


def prepare_model_inputs_and_targets(w_emb):
    """
        Prepare train and test <X,Y> for neural network

        :param1 w_emb: Word2Vec model.
    """
    train = load(c.TRAIN_CORPUS)
    test = load(c.TEST_CORPUS)

    targets_train = []
    sentences_train = []
    targets_test = []
    sentences_test = []

    ht_lists = []
    for tweet in train:
        ht_list = hashtags_list(tweet, w_emb)
        h_embedding = emb.tweet_arith_embedding(w_emb, " ".join(ht_list))
        if h_embedding is not None:
            targets_train.append(emb.np.array(h_embedding))
            sentences_train.append(tweet)
    for tweet in test:
        ht_list = hashtags_list(tweet, w_emb)
        h_embedding = emb.tweet_arith_embedding(w_emb, " ".join(ht_list))
        if h_embedding is not None:
            targets_test.append(h_embedding)
            sentences_test.append(tweet)
            ht_lists.append(ht_list)

    sentences_train_len = len(sentences_train)
    targets_train_len = len(targets_train)
    sentences = sentences_train
    sentences.extend(sentences_test)
    targets = targets_train
    targets.extend(targets_test)
    sentences, targets = _remove_hashtags_from_sentences(sentences, targets)

    targets_train = np.array(targets[:targets_train_len])
    targets_test = np.array(targets[targets_train_len:])

    sentences_train = np.array(sentences[:sentences_train_len])
    sentences_test = np.array(sentences[sentences_train_len:])

    return sentences_train, sentences_test, targets_train, targets_test, ht_lists
