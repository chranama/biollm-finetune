import json
import pandas as pd
import subprocess
import requests
import time

with open('BioASQ-training12b/training12b_new.json') as file:
    training_list = json.load(file)['questions']

training_df = pd.json_normalize(training_list)
training_yesno = training_df[training_df.type == 'yesno'].reset_index(drop=True)
text = training_yesno.body.iloc[0]

subprocess.run('chmod -x token/set_token.sh', shell=True)
subprocess.run('./token/set_token.sh', shell=True)
subprocess.run('docker run --gpus all -e HF_TOKEN=$HF_TOKEN --name model -p 8000:8000 ghcr.io/mistralai/mistral-src/vllm:latest --host 0.0.0.0 --model mistralai/Mistral-7B-Instruct-v0.2', shell=True)

parameters={'model':'mistral-large-latest', 'messages': [{'role': 'user', 'content': text}]}
start = time.time()
result = requests.get('http://localhost:8000', params=parameters)
end = time.time()

print(result)

print('The local API call took ' +str(end - start)+ ' seconds to execute.')

