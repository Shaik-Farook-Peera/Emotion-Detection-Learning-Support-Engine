import os
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "datasets",
    "train.txt"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "datasets",
    "test.txt"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "bert",
    "base"
)

os.makedirs(MODEL_PATH, exist_ok=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_dataset(path):

    texts = []
    emotions = []

    with open(path, "r", encoding="utf-8") as file:

        for line in file:

            text, emotion = line.strip().split(";")

            texts.append(text)
            emotions.append(emotion)

    return pd.DataFrame({
        "text": texts,
        "emotion": emotions
    })


train_df = load_dataset(TRAIN_PATH)
test_df = load_dataset(TEST_PATH)

# --------------------------------------------------
# Label Encoding
# --------------------------------------------------

encoder = LabelEncoder()

train_df["label"] = encoder.fit_transform(
    train_df["emotion"]
)

test_df["label"] = encoder.transform(
    test_df["emotion"]
)

# --------------------------------------------------
# HuggingFace Dataset
# --------------------------------------------------

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]]
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]]
)

# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(encoder.classes_)
)

# --------------------------------------------------
# Accuracy
# --------------------------------------------------

def compute_metrics(pred):

    predictions = pred.predictions.argmax(-1)

    return {
        "accuracy": accuracy_score(
            pred.label_ids,
            predictions
        )
    }

# --------------------------------------------------
# Training
# --------------------------------------------------

training_args = TrainingArguments(

    output_dir="./bert_output",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_steps=100,

    load_best_model_at_end=True
)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics
)

# --------------------------------------------------
# Train
# --------------------------------------------------

trainer.train()

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

result = trainer.evaluate()

print(result)

# --------------------------------------------------
# Save
# --------------------------------------------------

trainer.save_model(MODEL_PATH)

tokenizer.save_pretrained(MODEL_PATH)

print("\nBERT Model Saved Successfully")