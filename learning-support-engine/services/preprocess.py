import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


DATASET_PATH = Path("data/datasets")
MODELS_PATH = Path("models/bilstm")

MODELS_PATH.mkdir(parents=True, exist_ok=True)


def load_dataset(filename):
    data = []

    with open(DATASET_PATH / filename, "r", encoding="utf-8") as file:
        for line in file:
            text, emotion = line.strip().split(";")
            data.append({"text": text, "emotion": emotion})

    return pd.DataFrame(data)


train_df = load_dataset("train.txt")
test_df = load_dataset("test.txt")
val_df = load_dataset("val.txt")

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(train_df["emotion"])
y_test = label_encoder.transform(test_df["emotion"])
y_val = label_encoder.transform(val_df["emotion"])

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["text"])

X_train = tokenizer.texts_to_sequences(train_df["text"])
X_test = tokenizer.texts_to_sequences(test_df["text"])
X_val = tokenizer.texts_to_sequences(val_df["text"])

MAX_LENGTH = 100

X_train = pad_sequences(X_train, maxlen=MAX_LENGTH, padding="post")
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH, padding="post")
X_val = pad_sequences(X_val, maxlen=MAX_LENGTH, padding="post")

with open(MODELS_PATH / "tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

with open(MODELS_PATH / "label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

print("Training Shape :", X_train.shape)
print("Validation Shape :", X_val.shape)
print("Testing Shape :", X_test.shape)

print("Classes :", label_encoder.classes_)

print("Tokenizer Saved")
print("Label Encoder Saved")