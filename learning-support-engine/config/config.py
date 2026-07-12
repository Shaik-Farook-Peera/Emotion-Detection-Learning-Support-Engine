import os

from dotenv import load_dotenv

# ---------------------------------------
# Load Environment Variables
# ---------------------------------------

load_dotenv()

# ---------------------------------------
# Gemini API Key
# ---------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if __name__ == "__main__":

    if GEMINI_API_KEY:
        print("Gemini API Key Loaded Successfully.")
    else:
        print("Gemini API Key Not Found.")