import requests as rq
from datetime import datetime

BASE_URL = 'https://adminapi.pythonanywhere.com/'

payload = {'input': 'hello!'}
response = rq.get(BASE_URL, params=payload)

json_values = response.json()
