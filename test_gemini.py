import requests

API_KEY = "حطي هنا الـ API KEY بتاعك"

url = f"https://generativelanguage.googleapis.com/v1/models?key={API_KEY}"

r = requests.get(url)

print(r.status_code)
print(r.text)
