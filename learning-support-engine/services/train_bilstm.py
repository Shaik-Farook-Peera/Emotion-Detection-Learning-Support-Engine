import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATASET_PATH = Path("data/datasets")
MODEL_PATH = Path("models/bilstm")

MODEL_PATH.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_dataset(filename):

    data = []

    with open(DATASET_PATH / filename, "r", encoding="utf-8") as file:

        for line in file:

            text, emotion = line.strip().split(";")

            data.append({
                "text": text,
                "emotion": emotion
            })

    return pd.DataFrame(data)


train_df = load_dataset("train.txt")
test_df = load_dataset("test.txt")
val_df = load_dataset("val.txt")

# --------------------------------------------------
# Label Encoding
# --------------------------------------------------

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(train_df["emotion"])
y_test = label_encoder.transform(test_df["emotion"])
y_val = label_encoder.transform(val_df["emotion"])

# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = Tokenizer(
    num_words=10000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(train_df["text"])

X_train = tokenizer.texts_to_sequences(train_df["text"])
X_test = tokenizer.texts_to_sequences(test_df["text"])
X_val = tokenizer.texts_to_sequences(val_df["text"])

MAX_LENGTH = 100

X_train = pad_sequences(X_train, maxlen=MAX_LENGTH, padding="post")
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH, padding="post")
X_val = pad_sequences(X_val, maxlen=MAX_LENGTH, padding="post")

# --------------------------------------------------
# Save Tokenizer
# --------------------------------------------------

with open(MODEL_PATH / "tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

with open(MODEL_PATH / "label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

# --------------------------------------------------
# BiLSTM Model
# --------------------------------------------------

model = Sequential()

model.add(
    Embedding(
        input_dim=10000,
        output_dim=128,
        input_length=MAX_LENGTH
    )
)

model.add(
    Bidirectional(
        LSTM(64)
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# Train
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# --------------------------------------------------
# Test
# --------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f"\nTest Accuracy : {accuracy:.4f}")

# --------------------------------------------------
# Save Model
# --------------------------------------------------

model.save(
    MODEL_PATH / "model.keras"
)

print("\nBiLSTM Model Saved Successfully")