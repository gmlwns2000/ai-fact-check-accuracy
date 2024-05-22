import requests
import json
import os
import tqdm

total_length = 4700
chunk = 10

os.makedirs('data', exist_ok=True)

data = []
for i in tqdm.tqdm(range(0, total_length, chunk), dynamic_ncols=True):
    url = f'https://factcheck-api.com/facts?limit=10&offset={i}&checked=1&hidden=0'
    res = requests.get(url=url)
    data += res.json()

with open('./data/snu_fact.json', 'w') as f:
    json.dump(data, f, indent=2)