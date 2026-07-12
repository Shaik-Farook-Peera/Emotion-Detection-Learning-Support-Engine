import os
import pandas as pd


class CSVLogger:

    def __init__(self):

        self.file_path = "data/prediction_history.csv"

        os.makedirs("data", exist_ok=True)

        if not os.path.exists(self.file_path):

            df = pd.DataFrame(columns=[

                "text",
                "model",
                "primary_emotion",
                "primary_confidence",
                "secondary_emotions"

            ])

            df.to_csv(self.file_path, index=False)

    def save(self, text, prediction):

        secondary = ", ".join(

            emotion["emotion"]

            for emotion in prediction["secondary_emotions"]

        )

        row = {

            "text": text,

            "model": prediction["model"],

            "primary_emotion": prediction["primary_emotion"],

            "primary_confidence": prediction["primary_confidence"],

            "secondary_emotions": secondary

        }

        df = pd.read_csv(self.file_path)

        df = pd.concat(

            [df, pd.DataFrame([row])],

            ignore_index=True

        )

        df.to_csv(self.file_path, index=False)