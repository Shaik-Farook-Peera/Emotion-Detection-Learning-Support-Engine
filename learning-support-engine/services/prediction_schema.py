class PredictionSchema:

    @staticmethod
    def build(

        model_name,
        cleaned_text,

        primary_emotion,
        primary_confidence,

        secondary_emotions,

        scores

    ):

        return {

            "model": model_name,

            "cleaned_text": cleaned_text,

            "primary_emotion": primary_emotion,

            "primary_confidence": float(primary_confidence),

            "secondary_emotions": secondary_emotions,

            "scores": scores

        }