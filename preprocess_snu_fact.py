import json
import time
import pandas as pd
from datetime import datetime
import os

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
            'is_fact': s > 3,
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

def load_df_rag():
    root = './data/google'
    
    df = load_df()
    
    search_terms = []
    rags = []
    
    for i in range(len(df)):
        path = os.path.join(root, f'{i}.json')
        with open(path, 'r') as f:
            data = json.load(f)
        search_term = data['queries']['request'][0]['searchTerms']
        search_terms.append(search_term)
        rag = []
        if 'items' in data:
            for i, item in enumerate(data['items']):
                if 'snippet' in item:
                    rag.append(f'> 문서 {i}\n제목: {item["title"]}\n내용 요약: {item["snippet"]}')
                else:
                    rag.append(f'> 문서 {i}\n제목: {item["title"]}')
        else:
            rag.append('검색 결과 없음')
        search_result = '\n\n'.join(rag)
        search_result = f'=== 구글 검색 결과 (검색어: {search_term}) ===\n\n' + search_result
        rags.append(search_result)
    
    df['search_term'] = search_terms
    df['rag'] = rags
    return df

def load_df_est(model_name='qwen1.5_32b'):
    df = load_df_rag()
    
    truth = []
    votes = []
    avgs = []
    maxs = []
    firsts = []
    for i in range(len(df)):
        path = f'./data/result_{model_name}/{i}.json'
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            truth.append(data['truth'])
            votes.append(data['voted'])
            avgs.append(data['avg'])
            maxs.append(max(data['estimations']))
            firsts.append(data['estimations'][0])
        else:
            truth.append(None)
            votes.append(None)
            avgs.append(None)
            maxs.append(None)
            firsts.append(None)
    
    df['estimation_truth'] = truth
    df['estimation_average'] = avgs
    df['estimation_vote'] = votes
    df['estimation_max'] = maxs
    df['estimation_first'] = firsts
    
    return df

if __name__ == '__main__':
    print(load_df_est())