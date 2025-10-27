#!/usr/bin/python3

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import * ##DatasetDict, load_dataset
from datetime import datetime
##from utils.functions import *

torch.cuda.empty_cache()

######## Accelerator part

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)



## Load model ---------------------------

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
new_model = "Mistral-7B-Instruct-v0.1-biqa"
##finetuning_dataset = "one_example.jsonl"
finetuning_dataset = sys.argv[1]
project = sys.argv[2] ## extension to the model

## Load training dataset
full_dataset = load_dataset("json", data_files=finetuning_dataset, split="train")
full_dataset = full_dataset.remove_columns(['snippets', 'documents']) ## XXX removes those unused columns for now 
print("Full dataset loaded.")
#print(train_dataset)
#print(train_dataset["body"][0])
#print(train_dataset.shape)

filtered_dataset = full_dataset.filter(lambda example: example['type'] != "not_usable")


## Dataset split for training nad testing ----------------------------

full_dataset_split = filtered_dataset.train_test_split(test_size=0.2, seed=42)
print(full_dataset_split)


#####---------------- model initialization

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
    use_cache='False'
)


###### Tokenization

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_lenght=512,
    padding_side="left",
    add_bos_token=True,
)

tokenizer.pad_token = tokenizer.eos_token

# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

## Queries formating -------------------------------------------------

## used for the fine-tuning
## we give question and answer
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



max_length = 512 # differs from datasets to datasets

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
## checked and non-zero
# print(train_dataset)
# print(eval_dataset)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)




# ## model configuration LORA and perf -------------------------------------------

from peft import prepare_model_for_kbit_training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



from peft import LoraConfig, get_peft_model

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
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print("Trainable parameters: ")
print_trainable_parameters(model)

# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)


if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True



## model training step ----------------------------------------------

##project = "smallBiQA-finetune"
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
        ##gradient_checkpointing=True,
        max_steps=100, ## originally 500
        learning_rate=2.5e-4, # Want a small lr for finetuning
        #bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=25,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=25,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# # Save the fine-tuned model
# trainer.model.save_pretrained(new_model)

# Save the model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)




## OLD CODE ----------------------------------------------
## generate answers
## used for applying the model 

# def generate_yesno(question,model):
#     prompt = "### You can only use JSON format to answer my questions. The format must be {”exact_answer”:””, ”ideal_answer”:””}, where exact_answer should be 'yes' or 'no', and ideal_answer is a short conversational response starting with yes/no then follow on the explain. ### The question is:" + question 
#     encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
#     model_inputs = encoded_input.to('cuda')

#     generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

#     decoded_output = tokenizer.batch_decode(generated_ids)

#     return decoded_output[0].replace(prompt, "")


# def generate_list(question,model):
#     prompt = '### You can only use JSON format to answer my questions. The format must be {”exact_answer”:[], ”ideal_answer”:””}, where exact_answer is a list of precisekey entities to answer the question, and ideal_answer is a short conversational response containing an explanation. ### The question is:' + question
#     encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
#     model_inputs = encoded_input.to('cuda')

#     generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

#     decoded_output = tokenizer.batch_decode(generated_ids)

#     return decoded_output[0].replace(prompt, "")


# def generate_factoid(question,model):
#     prompt = '### You can only use JSON format to answer my questions. The format must be {”exact_answer”:[], ”ideal_answer”:””}, where exact_answer is a list of precise key entities to answer the question. ideal_answer is a short conversational response containing an explanation. ### The question is:' + question
#     encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
#     model_inputs = encoded_input.to('cuda')

#     generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

#     decoded_output = tokenizer.batch_decode(generated_ids)

#     return decoded_output[0].replace(prompt, "")


# def generate_summary(question,model):
#     prompt = "## Reply to the answer clearly and easily in less than 3 sentences. ### The question is:" + question

#     encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
#     model_inputs = encoded_input.to('cuda')

#     generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

#     decoded_output = tokenizer.batch_decode(generated_ids)

#     return decoded_output[0].replace(prompt, "")


# num_line = train_dataset.shape[0]
# for line in range(0, num_line):
#     question = train_dataset["body"][line]
#     print("Question:\n"+ prompt)

#     if train_dataset["type"][line] == "yesno":
#         print("Yes/no answer:")
#         ## testing other generate response types: 
#         print(generate_yesno(prompt, model))
#         print("\n")

#     elif train_dataset["type"][line] == "list":
#         print("List answer:")
#         print(generate_list(prompt, model))
#         print("\n")

#     elif train_dataset["type"][line] == "factoid":
#         print("Factoid answer:")
#         print(generate_factoid(prompt, model))
#         print("\n")

#     elif train_dataset["type"][line] == "summary":
#         print("Summary answer:")
#         print(generate_summary(prompt, model))
#         print("\n")

