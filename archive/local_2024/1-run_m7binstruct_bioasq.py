#!/usr/bin/python3

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset, load_dataset
##from utils.functions import *

## Load model ---------------------------

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
##finetuning_dataset = "one_example.jsonl"
finetuning_dataset = sys.argv[1]

## Load training dataset
train_dataset = load_dataset("json", data_files=finetuning_dataset, split="train")
print("training dataset loaded.")
#print(train_dataset)
#print(train_dataset["body"][0])
#print(train_dataset.shape)


## for bit model version - Uses less memory
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    quantization_config=nf4_config,
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


#example
def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids)

  return decoded_output[0].replace(prompt, "")

## first question in the dataset
##prompt =train_dataset["body"][0]
##print(generate_response(prompt, model))
##print(train_dataset.shape)






## generate answers
def generate_yesno(question,model):
    prompt = "### You can only use JSON format to answer my questions. The format must be {”exact_answer”:””, ”ideal_answer”:””}, where exact_answer should be 'yes' or 'no', and ideal_answer is a short conversational response starting with yes/no then follow on the explain. ### The question is:" + question 
    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')

    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    decoded_output = tokenizer.batch_decode(generated_ids)

    return decoded_output[0].replace(prompt, "")


def generate_list(question,model):
    prompt = '### You can only use JSON format to answer my questions. The format must be {”exact_answer”:[], ”ideal_answer”:””}, where exact_answer is a list of precisekey entities to answer the question, and ideal_answer is a short conversational response containing an explanation. ### The question is:' + question
    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')

    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    decoded_output = tokenizer.batch_decode(generated_ids)

    return decoded_output[0].replace(prompt, "")


def generate_factoid(question,model):
    prompt = '### You can only use JSON format to answer my questions. The format must be {”exact_answer”:[], ”ideal_answer”:””}, where exact_answer is a list of precise key entities to answer the question. ideal_answer is a short conversational response containing an explanation. ### The question is:' + question
    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')

    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    decoded_output = tokenizer.batch_decode(generated_ids)

    return decoded_output[0].replace(prompt, "")


def generate_summary(question,model):
    prompt = "## Reply to the answer clearly and easily in less than 3 sentences. ### The question is:" + question

    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')

    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    decoded_output = tokenizer.batch_decode(generated_ids)

    return decoded_output[0].replace(prompt, "")


num_line = train_dataset.shape[0]
for line in range(0, num_line):
    prompt = train_dataset["body"][line]
    print("Question:\n"+ prompt)

    if train_dataset["type"][line] == "yesno":
        print("Yes/no answer:")
        ## testing other generate response types: 
        print(generate_yesno(prompt, model))
        print("\n")

    elif train_dataset["type"][line] == "list":
        print("List answer:")
        print(generate_list(prompt, model))
        print("\n")

    elif train_dataset["type"][line] == "factoid":
        print("Factoid answer:")
        print(generate_factoid(prompt, model))
        print("\n")

    elif train_dataset["type"][line] == "summary":
        print("Summary answer:")
        print(generate_summary(prompt, model))
        print("\n")


## tested: 2023-03-25
## Running OK 