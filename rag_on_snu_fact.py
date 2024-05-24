import os
os.environ['ATTENTION_BACKEND'] = 'vllm'
import json
from preprocess_snu_fact import load_df_rag
import requests
import time
import urllib
import vllm
import transformers
from collections import Counter

model_name = 'qwen1.5_32b'

model_id = {
    'eeve_11b': 'yanolja/EEVE-Korean-Instruct-10.8B-v1.0',
    'qwen1.5_14b': 'Qwen/Qwen1.5-14B-Chat-GPTQ-Int4',
    'qwen1.5_32b': 'yanolja/EEVE-Korean-Instruct-10.8B-v1.0'    #  0.1465, acc 23.46, KOR
    # 'qwen1.5_32b': 'Qwen/Qwen1.5-14B-Chat-GPTQ-Int4',         #  0.1408, acc 20.46, ENG/CHN
    # 'qwen1.5_32b': 'Qwen/Qwen1.5-32B-Chat-AWQ',               #  0.1321, acc 13.11, ENG/CHN
    # 'qwen1.5_32b': 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int4',          # -0.0265, acc 18.00, ENG/CHN
    # 'qwen1.5_32b': 'chihoonlee10/T3Q-ko-solar-dpo-v7.0'       # -0.0491, acc 21.00, KOR
    # 'qwen1.5_32b': 'saltlux/Ko-Llama3-Luxia-8B'               # -0.1912, acc 18.66, KOR
}[model_name]

llm = vllm.LLM(
    model=model_id,
    max_model_len=4000,
    tensor_parallel_size=2,
    kv_cache_dtype='fp8_e5m2'
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

system_prompt = """
<|im_start|>system
당신은 최고의 추론가입니다.
당신은 구글 검색을 기반으로 주어진 주장이 사실인지 아닌지 적절히 판단해야 합니다.
당신은 정직해야합니다.
당신은 적절히 사실과 근거를 바탕으로 사실인지 아닌지 판단해야합니다.
당신은 주어진 사실을 5, 4, 3, 2, 1, 0, -1 중 하나의 숫자로 사실인지 아닌지 대답해야합니다. 
- 5는 매우 사실임을 나타냅니다.
- 4는 조금 사실임을 나타냅니다.
- 3은 모호하게 사실인것 같습니다.
- 2는 모호하게 거짓인것 같습니다.
- 1는 조금 거짓임을 나타냅니다.
- 0은 매우 거짓임을 나타냅니다.
- -1은 주장이 사실인지 거짓인지 나타낼 검색결과가 없는 경우입니다.
구글 검색 결과와 검색어를 잘 읽어 보고 생각해보세요.
만약 검색결과가 주장을 뒷받침 하는 근거를 포함하고 있지않다면 -1을 대답하세요.
주장에 대한 확실한 근거가 없다면 -1을 대답해야합니다.
이제부터 시작입니다.
<|im_end|>
"""

icl_prompt = """<|im_start|>user
주장: {claim}

{rag}
<|im_end|>
<|im_start|>assistant
{score}<|im_end|>
"""

prompt = """<|im_start|>user
주장: {claim}

{rag}
<|im_end|>
<|im_start|>assistant
"""

import random

def get_random_icl(df, n=1):
    length = len(df['rag'])
    text = ""
    for i in range(n):
        j = random.randint(0, length-1)
        text += icl_prompt.format(
            claim=df['message'][j],
            rag=df['rag'][j],
            score=df['score'][j],
        )
    return text

df = load_df_rag()

def estimate(df, i):
    claim = df['message'][i]
    rag = df['rag'][i]
    score = df['score'][i]
    
    sampling_params = vllm.SamplingParams(
        temperature=0.7,
        top_p=0.9, 
        top_k=1,
        max_tokens=1,
        stop=['<|im_start|>']
    )
    
    input = system_prompt + get_random_icl(df) + prompt.format(claim=claim, rag=rag)
    # input = system_prompt + prompt.format(claim=claim, rag=rag)
    
    # print(input)
    
    output = llm.generate(input, sampling_params, use_tqdm=False)[0]
    generated_text = output.outputs[0].text
    
    estimated_score = -1
    if '1' in generated_text:
        estimated_score = 1
    elif '2' in generated_text:
        estimated_score = 2
    elif '3' in generated_text:
        estimated_score = 3
    elif '4' in generated_text:
        estimated_score = 4
    elif '5' in generated_text:
        estimated_score = 5
    elif '0' in generated_text:
        estimated_score = 0
    print(i, score, generated_text, estimated_score)
    
    return score, estimated_score

def main():
    os.makedirs(f'./data/result_{model_name}', exist_ok=True)
    
    for i in range(len(df['message'])):
        estimations = []
        for _ in range(1):
            truth, estimation = estimate(df, i)
            estimations.append(estimation)
        counts = Counter(estimations)
        voted = counts.most_common(1)[0][0]
        
        print(truth, counts, voted)
        
        with open(f'data/result_{model_name}/{i}.json', 'w') as f:
            json.dump({
                'truth': int(truth),
                'voted': voted,
                'avg': sum(estimations) / len(estimations),
                'estimations': estimations,
            }, f, indent=2)

if __name__ == '__main__':
    main()