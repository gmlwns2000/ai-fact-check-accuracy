import json
import time
import pandas as pd
from datetime import datetime

def load_df(json_path = './data/snu_fact.json'):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print('loaded facts', len(data))

    unrolled_data = []

    for item in data:
        if not item['verified']:
            continue
        
        for i in range(len(item['scores'])):
            s = item['scores'][i]['score']
            if isinstance(s, int): break
        item_date = datetime.fromisoformat(item['createdAt'])
        m = item['lead_message']
        
        unrolled_data.append({
            'is_fact': s >= 4,
            'score': s,
            'issue_date': item_date,
            'message': m,
        })
    
    unrolled_data = list(sorted(unrolled_data, key=lambda x: x['issue_date']))
    
    is_fact = list(map(lambda x: x['is_fact'], unrolled_data))
    fact_score = list(map(lambda x: x['score'], unrolled_data))
    issue_date = list(map(lambda x: x['issue_date'], unrolled_data))
    message = list(map(lambda x: x['message'], unrolled_data))

    df = pd.DataFrame()
    df['is_fact'] = is_fact
    df['score'] = fact_score
    df['issue_date'] = issue_date
    df['message'] = message
    
    return df

if __name__ == '__main__':
    print(load_df())