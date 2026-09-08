import httpx
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

r = httpx.get(url)

print(r.status_code)
print(r.text)