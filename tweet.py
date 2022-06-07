from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class Tweet:
    original_tweet: Dict[str, Any]
    text: str
    _hashtags: List[str] = None

    @property
    def hashtags(self) -> List[str]:
        if self._hashtags is None:
            self._hashtags = [
                '#' + hashtag_text for hashtag_text in self.original_tweet['hashtags']
            ]

        return self._hashtags

    @hashtags.setter
    def hashtags(self, hashtags: List[str]) -> None:
        self._hashtags = hashtags

    @property
    def tokens(self) -> List[str]:
        return self.text.split()

    def __str__(self):
        return (
            f'original text: {self.original_tweet["text"]}\n'
            f'text: {self.text}\n'
            f'original tweet: {self.original_tweet}'
        )
