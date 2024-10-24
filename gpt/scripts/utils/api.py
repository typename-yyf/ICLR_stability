import requests
from data_utils import Tokenized_data
import time
import csv

proxy_url = 'http://api.52099520.xyz/v1'
api_key = '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'
# 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

def get_gpt_response():
    dataset = Tokenized_data(['legal', 'review'], 256, is_test=True)
    f = open('embeddings.txt', 'w')
    csv_writer = csv.writer(f)
    for index in range(10):
        domain_idx = index % len(dataset.domain_texts)
        index = index // len(dataset.domain_texts)
        input = dataset.domain_texts[domain_idx][index]
        params = {
            'model': 'text-embedding-ada-002',
            'input': input,
        }
        
        response = requests.post(f'{proxy_url}/embeddings', json=params,  auth=('', api_key))
        response = response.json()
        
        res = response['data'][0]['embedding']
        csv_writer.writerow(res)
        for sec in range(30):
            print(f'processed {index}, waiting{30-sec}...    ', end='\r')
            time.sleep(1)


if __name__ == '__main__':
    get_gpt_response()