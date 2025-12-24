import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GOOGLE_GEMINI_MODEL_NAME = os.getenv("GOOGLE_GEMINI_MODEL_NAME", "gemini-2.5-flash")
