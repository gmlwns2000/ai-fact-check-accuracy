import os
os.environ['ATTENTION_BACKEND'] = 'vllm'
import json
from preprocess_snu_fact import load_df
import requests
import time
import urllib
from secret import *
import vllm

llm = vllm.LLM(
    model='yanolja/EEVE-Korean-Instruct-10.8B-v1.0',
    max_model_len=8000,
    tensor_parallel_size=2,
)

df = load_df()

root = './data/google'

os.makedirs(root, exist_ok=True)

for i in range(len(df)):
    path = os.path.join(root, f'{i}.json')
    if os.path.exists(path):
        continue
    
    keyword = df['message'][i].strip().replace('는 사실?', '').replace('"', '').replace('“', "").replace('”', '')
    
    prompt = f"<|im_start|>system\n당신은 어떠한 주장을 위해 자료를 찾으려 구글에 검색을 해야한다. \n 구글에 검색할 키워드를 만드는 전문가이다. \n 당신은 한국어만을 사용한다 \n 당신은 중국어를 쓰면 안된다.<|im_end|>\n<|im_start|>user\n다음 주장을 증명할 자료를 찾는데 필요한 구글링 검색어를 만들어라. 검색어만 답장하라. 검색어를 제외한 다른것은 절대 대답하지말아라. 따옴표로 감싸지 말아라. 가장 좋은 하나만 답하라\n\n주장: {keyword}\n\n<|im_end|>\n<|im_start|>assistant\n구글 검색 키워드: "
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=30)
    output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
    generated_text = output.outputs[0].text.replace('"', '').replace("\n", '')
    print(keyword, '-->', generated_text)
    keyword = generated_text
    
    url = f'https://www.googleapis.com/customsearch/v1?key={GOOGLE_SEARCH_KEY}&cx={GOOGLE_SEARCH_CX}&gl=kr&hl=kr&lr=lang_ko&num=10&q={urllib.parse.quote_plus(keyword)}'
    print(f'"""{keyword}"""', url)
    
    res = requests.get(url)
    res_json = res.json()
    
    print(res.status_code)
    
    assert res.status_code == 200, res.text
    
    if 'items' in res_json:
        print(len(res_json['items']), 'results')
    else:
        print('no results')
    
    with open(path, 'w', encoding='UTF-8') as f:
        json.dump(res_json, f, indent=2, ensure_ascii=False)
    
    time.sleep(1.0)

print('done')