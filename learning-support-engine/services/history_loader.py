import pandas as pd
import os


class HistoryLoader:

    FILE = "data/interaction_history.csv"

    @staticmethod
    def load():

        if not os.path.exists(HistoryLoader.FILE):
            return pd.DataFrame()

        return pd.read_csv(HistoryLoader.FILE)