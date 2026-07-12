import re
import nltk

from nltk.tokenize import word_tokenize
from services.keyword_boost import EMOTION_KEYWORDS

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class TextPreprocessor:

    def __init__(self):

        self.skip_words = {
            "the",
            "a",
            "an"
        }

    def clean_text(self, text: str) -> str:

        text = str(text).lower()

        # Keep letters, spaces, apostrophes, ! and ?
        text = re.sub(r"[^a-zA-Z!?'\s]", " ", text)

        tokens = word_tokenize(text)

        tokens = [
            token
            for token in tokens
            if token not in self.skip_words
            and token.strip()
        ]

        cleaned = " ".join(tokens)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def keyword_scores(self, text):

        tokens = text.split()

        scores = {}

        for emotion, words in EMOTION_KEYWORDS.items():

            score = 0

            for word in words:

                if word in tokens:
                    score += 10

            scores[emotion] = score

        return scores