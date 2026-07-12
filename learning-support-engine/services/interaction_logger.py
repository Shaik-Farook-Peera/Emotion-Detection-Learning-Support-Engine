import os
import pandas as pd
from datetime import datetime


class InteractionLogger:

    def __init__(self):

        self.file_path = "data/interaction_history.csv"

        os.makedirs("data", exist_ok=True)

        if not os.path.exists(self.file_path):

            df = pd.DataFrame(columns=[

                "timestamp",
                "field",
                "problem",
                "cleaned_text",
                "primary_emotion",
                "primary_confidence",
                "secondary_emotions",
                "bilstm_scores",
                "bert_emotion",
                "bert_confidence",
                "bert_scores",
                "ai_enabled",
                "response"

            ])

            df.to_csv(self.file_path, index=False)

    def save(self, prediction=None, **kwargs):

        if prediction is None:
            prediction = kwargs
        elif kwargs:
            prediction = {**prediction, **kwargs}

        if "primary_emotion" not in prediction and "emotion" in prediction:
            prediction["primary_emotion"] = prediction["emotion"]

        if "primary_confidence" not in prediction and "confidence" in prediction:
            prediction["primary_confidence"] = prediction["confidence"]

        if "response" not in prediction and "ai_response" in prediction:
            prediction["response"] = prediction["ai_response"]

        row = {

            "timestamp": datetime.now(),

            "field": prediction.get("field"),

            "problem": prediction.get("problem"),

            "cleaned_text": prediction.get("cleaned_text"),

            "primary_emotion": prediction.get("primary_emotion"),

            "primary_confidence": prediction.get("primary_confidence"),

            "secondary_emotions": prediction.get("secondary_emotions", "[]"),

            "bilstm_scores": prediction.get("bilstm_scores", "{}"),

            "bert_emotion": prediction.get("bert_emotion"),

            "bert_confidence": prediction.get("bert_confidence"),

            "bert_scores": prediction.get("bert_scores", "{}"),

            "ai_enabled": prediction.get("ai_enabled"),

            "response": prediction.get("response", prediction.get("ai_response"))

        }

        df = pd.read_csv(self.file_path)

        df = pd.concat(

            [df, pd.DataFrame([row])],

            ignore_index=True

        )

        df.to_csv(self.file_path, index=False)