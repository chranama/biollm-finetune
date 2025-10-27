#!/usr/bin/python3

import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, BitsAndBytesConfig, Trainer
from datasets import Dataset
import datasets
import torch
import pandas as pd
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import json
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import HfApi, HfFolder

torch.cuda.empty_cache()

hf_token = HfFolder.get_token()

api = HfApi()
user = api.whoami(hf_token)
print("Logged in as user:", user['name'])

finetuning_dataset = sys.argv[1]
project = sys.argv[2]

with open(finetuning_dataset) as file:
    lst = json.load(file)['questions']

df = pd.json_normalize(lst)

def flatten_list(lst):
    flat_list = []
    if type(lst) == float:
        lst = [lst]
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def extract_text(lst_of_dcts):
    text_list = []
    for item in lst_of_dcts:
        text_list.append(item.get('text'))
    return text_list

df['exact_answer'] = df['exact_answer'].apply(flatten_list)
df['snippets_text'] = df['snippets'].apply(extract_text)

dataset = Dataset.from_pandas(df)
#dataset = dataset.remove_columns(['documents', 'concepts', 'id', 'snippets'])
dataset = dataset.train_test_split(test_size=0.2)

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
new_model = 'Mistral-7B-Instruct-v0.1-BioASQdata-finetuning'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    quantization_config=nf4_config,
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=512,
    padding_side="left",
    add_bos_token=True,
)

tokenizer.pad_token = tokenizer.eos_token


def formatting_yesno(question, exact_answer, ideal_answer, snippets_text):
    text = "### This is an example of yes/no question and the respective answer in the intended format. Where the answer can only use JSON format. The format must be {”exact_answer”:””, ”ideal_answer”:””}, where exact_answer should be 'yes' or 'no', and ideal_answer is a short conversational response starting with yes/no then follow on the explain. \n### Yes/No question:" + question + '\n### context: ' + ' '.join(snippets_text) + '\n### Answer: {”exact_answer”:” ' + ' '.join(exact_answer) + '", ”ideal_answer”:”' + ' '.join(ideal_answer) + '"}"'
    return text

def formatting_list(question, exact_answer, ideal_answer, snippets_text):
    text ="### This is an example of list question and the respective answer in the intended format. Where the answer can only use JSON format. The format must be {”exact_answer”:[], ”ideal_answer”:””}, where exact_answer is a list of precisekey entities to answer the question, and ideal_answer is a short conversational response containing an explanation. \n### List question:" + str(question) + '\n### context: ' + ' '.join(snippets_text) + '\n### Answer: {”exact_answer”:” ' + ' '.join(exact_answer) + '", ”ideal_answer”:”' + ' '.join(ideal_answer) + '”}'
    return text

def formatting_factoid(question, exact_answer, ideal_answer, snippets_text):
    text = "### This is an example of factoid question and the respective answer in the intended format. Where the answer can only use JSON format. The format must be {”exact_answer”:[], ”ideal_answer”:””}, where exact_answer is a list of precise key entities to answer the question. ideal_answer is a short conversational response containing an explanation. \n### Factoid question:" + str(question) + '\n### context: ' + ' '.join(snippets_text) + '\n### Answer: {”exact_answer”:” ' + ' '.join(exact_answer) + '", ”ideal_answer”:”' + ' '.join(ideal_answer) + '”}'
    return text

def formatting_summary(question, exact_answer, ideal_answer, snippets_text):
    text = "### This is an example of summary question and the respective answer in the intended format. Where the reply to the answer clearly and easily in less than 3 sentences. \n### Summary question:" + question + '\n### context: ' + ' '.join(snippets_text) + "\n### Answer:" + ' '.join(ideal_answer)
    return text

max_length = 512

def generate_and_tokenize_prompt(learning_line):
    question = learning_line['body']
    question_type = learning_line['type']
    exact_answer = learning_line['exact_answer']
    ideal_answer = learning_line['ideal_answer']
    snippets_text = learning_line['snippets_text']

    if question_type == "yesno":
        result = tokenizer(
            formatting_yesno(question, exact_answer, ideal_answer, snippets_text),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    elif question_type == "list":
        result = tokenizer(
            formatting_list(question, exact_answer, ideal_answer, snippets_text),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    elif question_type == "factoid":
        result = tokenizer(
            formatting_factoid(question, exact_answer, ideal_answer, snippets_text),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    elif question_type == "summary":
        result = tokenizer(
            formatting_summary(question, exact_answer, ideal_answer, snippets_text),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    result["labels"] = result["input_ids"].copy()
    return result

train_dataset = dataset['train']
eval_dataset = dataset['test']

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print("Trainable parameters: ")
print_trainable_parameters(model)

# accelerator handles model and dataset preparation
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
model = accelerator.prepare(model)
tokenized_train_dataset = accelerator.prepare(tokenized_train_dataset)
tokenized_val_dataset = accelerator.prepare(tokenized_val_dataset)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

base_model_name = "mistral7Binstruct"
run_name = base_model_name + "-" + project
output_dir = run_name

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=TrainingArguments(
        output_dir=output_dir,
        warmup_steps=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=500,
        learning_rate=2.5e-4,
        logging_steps=100,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        do_eval=True,
        optim="paged_adamw_8bit"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
