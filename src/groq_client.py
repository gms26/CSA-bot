import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to the .env file."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

MODEL_NAME = "openai/gpt-oss-20b"