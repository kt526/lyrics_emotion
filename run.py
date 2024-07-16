import requests as rq
from datetime import datetime

BASE_URL = 'https://adminapi.pythonanywhere.com/'

payload = {'input': 'hello!'}
response = rq.get(BASE_URL, params=payload)

json_values = response.json()

rq_input = json_values['input']
timestamp = json_values['timestamp']
character_count = json_values['character_count']

print(f'Input is: {rq_input}')
print(f'Date is: {timestamp}')
print(f'Character count is: {character_count}')