import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from services.preprocessing import TextPreprocessor

MAX_LEN = 100


class BiLSTMClassifier:

    def __init__(self):

        print("Loading BiLSTM Model...")

        self.model = tf.keras.models.load_model(
            "models/bilstm/bilstm_student_adaptive.keras",
            compile=False
        )

        with open("models/bilstm/tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        with open("models/bilstm/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        self.preprocessor = TextPreprocessor()

        print("BiLSTM Loaded Successfully")

    def predict(self, text):

        cleaned_text = self.preprocessor.clean_text(text)

        sequence = self.tokenizer.texts_to_sequences([cleaned_text])

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        probabilities = self.model.predict(
            padded,
            verbose=0
        )[0]

        predicted_index = np.argmax(probabilities)

        predicted_emotion = self.label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = float(probabilities[predicted_index])

        scores = {}

        for i, emotion in enumerate(self.label_encoder.classes_):
            scores[emotion] = float(probabilities[i])

        return {

            "emotion": predicted_emotion,

            "confidence": confidence,

            "scores": scores,

            "cleaned_text": cleaned_text

        }