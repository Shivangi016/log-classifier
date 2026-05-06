# app/utils.py
# This file loads your secret keys from the .env file
# into Python's environment so other files can read them.

from dotenv import load_dotenv

load_dotenv()  # Reads .env and loads GROQ_API_KEY into environment