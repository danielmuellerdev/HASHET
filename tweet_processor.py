from typing import List
from pathlib import Path
import re
import json
from collections import Counter

import spacy
from spacy.tokens.token import Token as SpacyToken
import unidecode
import gensim

from tweet import Tweet
import constants as C


class TweetProcessor:
    URL_REGEX = r"@\w*|https?:?\/?\/?[\w.\/]*|https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"
    USER_NAMES_REGEX = re.compile(r' ?@\S+')
    ACRONYM_REGEX = re.compile(r'[A-Z]{2,5}$')
    HASHTAG_REGEX = re.compile(r' ?#\S+')
    WHITE_LIST = re.compile(r'[^\w+_ #]+')
    WORDS_TO_REMOVE = ['rt', 'ht', 'htt', 'https', 'http', 'https t']

    @staticmethod
    def _read_tweets_from_file(file_path: Path, filter_out_retweets: bool = True) -> List[Tweet]:
        tweets = []
        with open(file_path, encoding='utf-8') as file:
            tweet_dicts = [json.loads(line.strip()) for line in file]

        for tweet_dict in tweet_dicts:
            is_retweet = tweet_dict['quoted_tweet_id'] is not None
               
            # skip tweet without text and retweets (no benefits for the embedding phase)
            if len(tweet_dict['text']) == 0 or (filter_out_retweets and is_retweet):
                continue

            tweets.append(Tweet(original_tweet=tweet_dict, text=tweet_dict['text']))

        return tweets

    @staticmethod
    def _is_acronym(text: str) -> str:
        return TweetProcessor.ACRONYM_REGEX.match(text) is not None

    @staticmethod
    def _remove_user_names(text: str) -> str:
        return re.sub(TweetProcessor.USER_NAMES_REGEX, '', text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        return re.sub(TweetProcessor.URL_REGEX, '', text).strip()

    @staticmethod
    def _handle_accented_characters(accented_string: str) -> str:
        """ Replace accented characters with unaccented ones: 'ü' -> 'ue'. """
        unaccented_string = unidecode.unidecode(accented_string)
        return unaccented_string

    @staticmethod
    def _to_lowercase(text: str, exclude_acronyms: bool = False) -> str:
        """ Lowercases the entire input text except acronyms. """
        lowercased_tokens = []
        for token in text.split():
            if exclude_acronyms:
                lowercased_token = token.lower() if not TweetProcessor._is_acronym(token) else token
            else:
                lowercased_token = token.lower()
            lowercased_tokens.append(lowercased_token)
        
        return ' '.join(lowercased_tokens)

    @staticmethod
    def _remove_hashtags(text: str) -> str:
        return re.sub(TweetProcessor.HASHTAG_REGEX, '', text)

    @staticmethod
    def _only_keep_letters_spaces_underscores_hashtags(text: str) -> str:
        return re.sub(TweetProcessor.WHITE_LIST, '', text).strip()

    @staticmethod
    def _add_common_ngrams(tweets: List[Tweet]) -> None:
        ngrams = gensim.models.Phrases([tweet.tokens for tweet in tweets])
        for tweet in tweets:
            tweet_tokens_with_ngrams = ngrams[tweet.tokens]
            tweet.text = ' '.join(tweet_tokens_with_ngrams)

    @staticmethod
    def _remove_stopwords(tokens: List[SpacyToken]) -> List[SpacyToken]:
        return [token for token in tokens if not token.is_stop and token.text not in TweetProcessor.WORDS_TO_REMOVE]

    @staticmethod
    def _lemmatize(tokens: List[SpacyToken], ignore_acronyms: bool = True) -> List[SpacyToken]:
        lemmatized_tokens = []
        for token in tokens:
            if ignore_acronyms and TweetProcessor._is_acronym(token.text):
                lemmatized_tokens.append(token.text)
            else:
                lemmatized_tokens.append(token.lemma_)

        return lemmatized_tokens

    @staticmethod
    def _remove_stopwords_and_lemmatize(tweets: List[Tweet], spacy_batch_size: int = 100) -> None:
        nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

        # Taking advantage of spaCy .pipe() attribute to speed-up the cleaning process:
        texts = [tweet.text for tweet in tweets]
        for i, doc in enumerate(nlp.pipe(texts, batch_size=spacy_batch_size, n_process=4)):
            tokens = list(doc)
            tokens = TweetProcessor._remove_stopwords(tokens)
            tokens = TweetProcessor._lemmatize(tokens)

            cleaned_tweet = ' '.join(tokens)
            tweets[i].text = cleaned_tweet

            if i % (spacy_batch_size * 20) == 0:
                print(f'Removing stopwords and lemmatizing with spacy: [{i} / {len(tweets)}]')

    @staticmethod
    def _remove_tweets_without_hashtags(tweets: List[Tweet]) -> List[Tweet]:
        # NOTE: our dataset contains no tweets without hashtags, but other datasets might
        return [tweet for tweet in tweets if len(tweet.hashtags) > 0]

    @staticmethod
    def _remove_single_word_tweets(tweets: List[Tweet]) -> List[Tweet]: 
        # NOTE: our dataset contains 445 single word tweets
        return [tweet for tweet in tweets if len(tweet.tokens) > 1]

    @staticmethod
    def _remove_rare_hashtags_from_tweets(tweets: List[Tweet], hashtag_min_count: int = C.HASHTAG_MINCOUNT) -> None:
        hashtag_count = Counter(hashtag for tweet in tweets for hashtag in tweet.hashtags)

        for tweet in tweets:
            tweet.hashtags = [
                hashtag for hashtag in tweet.hashtags
                if hashtag_count[hashtag] >= hashtag_min_count
            ]

    def preprocess_tweets_for_word_embedding(self, file_path: Path) -> List[Tweet]:
        tweet_corpus = self._read_tweets_from_file(file_path, filter_out_retweets=False)

        for tweet in tweet_corpus:
            tweet.text = self._remove_urls(tweet.text)
            tweet.text = self._remove_user_names(tweet.text)
            tweet.text = self._handle_accented_characters(tweet.text)
            tweet.text = self._to_lowercase(tweet.text, exclude_acronyms=True)
            tweet.text = self._only_keep_letters_spaces_underscores_hashtags(tweet.text)

        self._add_common_ngrams(tweet_corpus)
        self._remove_stopwords_and_lemmatize(tweet_corpus)

        return tweet_corpus

    def preprocess_tweets_for_sentence_embedding(self, file_path: Path) -> List[Tweet]:
        """ Keeps punctuation, emoij's, stopwords and doesn't lemmatize as keeping the tweets mostly in their original form
        can provide valuable information for the BERT model. It seems that heavily preprocessing the input data to the BERT
        model can result in similar and in some cases worse performance than keeping the data mostly as is. """
        
        tweets = self._read_tweets_from_file(file_path, filter_out_retweets=True)

        tweets = self._remove_tweets_without_hashtags(tweets)
        tweets = self._remove_single_word_tweets(tweets)
        self._remove_rare_hashtags_from_tweets(tweets)

        for tweet in tweets:
            tweet.text = self._remove_urls(tweet.text)
            tweet.text = self._remove_hashtags(tweet.text)
            tweet.text = self._remove_user_names(tweet.text)
            tweet.text = self._handle_accented_characters(tweet.text)
            # have to lowercase the tweet since the BERT model is uncased
            tweet.text = self._to_lowercase(tweet.text)

        # NOTE: not necessarily needed, test whether this improves performance
        self._add_common_ngrams(tweets)

        return tweets

        
