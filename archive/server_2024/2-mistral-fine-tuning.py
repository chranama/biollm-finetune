#!/usr/bin/python3

import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, BitsAndBytesConfig, Trainer
from datasets import Dataset, load_dataset
import datasets
import torch
import pandas as pd
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import json
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import login
import os

torch.cuda.empty_cache()

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

login(token=hf_token)

# accelerator handles model and dataset preparation
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


finetuning_dataset = sys.argv[1]
model_path = sys.argv[2]
project = sys.argv[3]

full_dataset = load_dataset("json", data_files=finetuning_dataset, split="train")


full_dataset_split = full_dataset.train_test_split(test_size=0.2, seed=42)

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
new_model = 'Mistral-7B-Instruct-v0.1-BioASQdata-finetuning'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    quantization_config=nf4_config,
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    model_max_length=512,
    padding_side="left",
    add_bos_token=True,
)

tokenizer.pad_token = tokenizer.eos_token


def formatting_yesno(question, exact_answer, ideal_answer):
    text = "### This is an example of yes/no question and the respective answer in the intended format. Where the answer can only use JSON format. The format must be {”exact_answer”:””, ”ideal_answer”:””}, where exact_answer should be 'yes' or 'no', and ideal_answer is a short conversational response starting with yes/no then follow on the explain. \n### Yes/No question:" + question + '\n### Answer: {”exact_answer”:” ' + exact_answer + '", ”ideal_answer”:”' + ideal_answer + '"}"'
    return text


def formatting_list(question, exact_answer, ideal_answer):
    text ="### This is an example of list question and the respective answer in the intended format. Where the answer can only use JSON format. The format must be {”exact_answer”:[], ”ideal_answer”:””}, where exact_answer is a list of precisekey entities to answer the question, and ideal_answer is a short conversational response containing an explanation. \n### List question:" + str(question) + '\n### Answer: {”exact_answer”:” ' + exact_answer + '", ”ideal_answer”:”' + ideal_answer + '”}'
    return text

def formatting_factoid(question, exact_answer, ideal_answer):
    text = "### This is an example of factoid question and the respective answer in the intended format. Where the answer can only use JSON format. The format must be {”exact_answer”:[], ”ideal_answer”:””}, where exact_answer is a list of precise key entities to answer the question. ideal_answer is a short conversational response containing an explanation. \n### Factoid question:" + str(question) + '\n### Answer: {”exact_answer”:” ' + exact_answer + '", ”ideal_answer”:”' + ideal_answer + '”}'
    return text

def formatting_summary(question, exact_answer, ideal_answer):
    text = "### This is an example of summary question and the respective answer in the intended format. Where the reply to the answer clearly and easily in less than 3 sentences. \n### Summary question:" + question + "\n### Answer:" + ideal_answer
    return text

max_length = 512

def generate_and_tokenize_prompt(learning_line):
    ## Get the informaiton to build the pompt
    question = learning_line['body']
    question_type = learning_line['type']
    exact_answer =learning_line['exact_answer']
    ideal_answer = learning_line['ideal_answer']


    if question_type == "yesno":

        result = tokenizer(
            formatting_yesno(question, exact_answer, ideal_answer),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    elif question_type == "list":

        result = tokenizer(
            formatting_list(question, exact_answer, ideal_answer),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    elif question_type == "factoid":

        result = tokenizer(
            formatting_factoid(question, exact_answer, ideal_answer),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    elif question_type == "summary":

        result = tokenizer(
            formatting_summary(question, exact_answer, ideal_answer),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    result["labels"] = result["input_ids"].copy()
    return result

train_dataset = full_dataset_split['train']
eval_dataset = full_dataset_split['test']

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

model = accelerator.prepare_model(model)

#if torch.cuda.device_count() > 1: # If more than 1 GPU
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=500, #ajustar, 
        learning_rate=2.5e-4,
        logging_steps=100,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        do_eval=True,
        optim="paged_adamw_8bit",
        fp16=True,  # Enable mixed precision training
        ddp_find_unused_parameters=False
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
