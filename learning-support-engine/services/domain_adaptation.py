import os
import pickle
import pandas as pd
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "bilstm", "model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "bilstm", "tokenizer.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "models", "bilstm", "label_encoder.pkl")

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "corpus",
    "student_emotion_data.csv"
)

SAVE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "bilstm",
    "bilstm_student_adaptive.keras"
)

print("Loading base model...")

base_model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print("Loading student dataset...")

df = pd.read_csv(DATA_PATH)

X = tokenizer.texts_to_sequences(df["text"])

X = pad_sequences(
    X,
    maxlen=100,
    padding="post",
    truncating="post"
)

y = label_encoder.transform(df["emotion"])

print("Creating adaptive model...")

adaptive_model = clone_model(base_model)
adaptive_model.set_weights(base_model.get_weights())

adaptive_model.layers[0].trainable = False

adaptive_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callback = EarlyStopping(
    monitor="loss",
    patience=2,
    restore_best_weights=True
)

adaptive_model.fit(
    X,
    y,
    epochs=8,
    batch_size=4,
    callbacks=[callback],
    verbose=1
)

adaptive_model.save(SAVE_PATH)

print("\nAdaptive Model Saved")
print(SAVE_PATH)