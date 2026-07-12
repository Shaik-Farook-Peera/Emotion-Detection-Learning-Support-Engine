import os

import google.generativeai as genai

from dotenv import load_dotenv

from services.response_templates import EMOTION_RESPONSES

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def build_prompt(field, problem, emotion, confidence):

    return f"""
You are an AI Learning Support Assistant.

Student Field:
{field}

Problem:
{problem}

Detected Emotion:
{emotion}

Confidence:
{confidence:.2%}

Provide:

1. Acknowledge the student's emotion.
2. Explain the concept clearly.
3. Give 3 practical study tips.
4. End with motivation.

Keep the response friendly and concise.
"""

def get_gemini_response(field, problem, emotion, confidence):

    try:

        prompt = build_prompt(
            field,
            problem,
            emotion,
            confidence
        )

        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception:

        template = EMOTION_RESPONSES.get(
            emotion.lower(),
            {
                "emoji": "🙂",
                "response": "Keep learning.",
                "action": "Practice regularly."
            }
        )

        return (
            f"{template['emoji']} {template['response']}\n\n"
            f"Suggestion: {template['action']}"
        )