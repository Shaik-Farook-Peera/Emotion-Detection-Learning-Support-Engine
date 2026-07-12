import torch
import numpy as np

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from services.preprocessing import TextPreprocessor
from services.keyword_boost import EMOTION_KEYWORDS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERTClassifier:

    def __init__(self):

        print("Loading BERT Model...")

        self.tokenizer = BertTokenizer.from_pretrained(
            "models/bert/base"
        )

        self.model = BertForSequenceClassification.from_pretrained(
            "models/bert/base"
        )

        self.model.to(DEVICE)

        self.model.eval()

        self.preprocessor = TextPreprocessor()

        self.labels = [
            "sadness",
            "joy",
            "love",
            "anger",
            "fear",
            "surprise"
        ]

        print("BERT Loaded Successfully")

    def keyword_adjustment(self, text, probabilities):

        tokens = text.split()

        for emotion, words in EMOTION_KEYWORDS.items():

            if emotion not in self.labels:
                continue

            index = self.labels.index(emotion)

            for word in words:

                if word in tokens:
                    probabilities[index] += 0.10

        probabilities = probabilities / probabilities.sum()

        return probabilities

    def predict(self, text):

        cleaned = self.preprocessor.clean_text(text)

        encoded = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = self.model(**encoded)

            logits = outputs.logits

            probabilities = torch.softmax(
                logits,
                dim=1
            ).cpu().numpy()[0]

        probabilities = self.keyword_adjustment(
            cleaned,
            probabilities
        )

        prediction = np.argmax(probabilities)

        emotion = self.labels[prediction]

        confidence = float(probabilities[prediction])

        scores = {}

        for i, label in enumerate(self.labels):
            scores[label] = float(probabilities[i])

        return {

            "emotion": emotion,

            "confidence": confidence,

            "scores": scores,

            "cleaned_text": cleaned

        }