class MixedEmotionDetector:

    def __init__(self, threshold=0.15):
        self.threshold = threshold

    def detect(self, prediction):

        scores = prediction["scores"]

        primary_emotion = prediction["emotion"]

        primary_confidence = prediction["confidence"]

        secondary_emotions = []

        for emotion, score in scores.items():

            if emotion == primary_emotion:
                continue

            if score >= self.threshold:

                secondary_emotions.append({

                    "emotion": emotion,

                    "confidence": float(score)

                })

        secondary_emotions.sort(

            key=lambda x: x["confidence"],

            reverse=True

        )

        return {

            "primary_emotion": primary_emotion,

            "primary_confidence": primary_confidence,

            "secondary_emotions": secondary_emotions,

            "scores": scores,

            "cleaned_text": prediction["cleaned_text"]

        }