#!/usr/bin/env python3

import sys
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from huggingface_hub import login

def main():
    # ==============================
    # 1. Argument Parsing
    # ==============================
    if len(sys.argv) != 4:
        print("Usage: python fine_tune.py <training_data.jsonl> <model_checkpoint> <project_name>")
        sys.exit(1)

    training_data_path = sys.argv[1]  # e.g., 'drugbank_fulldb_summary.jsonl'
    model_checkpoint = sys.argv[2]    # e.g., 'mistral7Binstruct-2-goterms'
    project_name = sys.argv[3]        # e.g., '3-drugbank'

    # ==============================
    # 2. Environment Setup
    # ==============================
    # Ensure CUDA 11.8 compatibility
    assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation."

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # ==============================
    # 3. Hugging Face Authentication
    # ==============================
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HUGGINGFACE_HUB_TOKEN not found in environment variables.")

    # ==============================
    # 4. Load and Split Dataset
    # ==============================
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=training_data_path, split="train")

    print("Splitting dataset into train and test...")
    dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # ==============================
    # 5. Initialize BitsAndBytesConfig
    # ==============================
    print("Configuring BitsAndBytes for 4-bit quantization...")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # ==============================
    # 6. Load Model and Tokenizer
    # ==============================
    print(f"Loading model from checkpoint: {model_checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        device_map='auto',
        quantization_config=nf4_config,
        use_cache=False
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        model_max_length=512,
        padding_side="left",
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ==============================
    # 7. Prepare Model for PEFT
    # ==============================
    print("Preparing model for k-bit training with PEFT...")
    model = prepare_model_for_kbit_training(model)

    # ==============================
    # 8. Define PEFT Configuration (LoRA)
    # ==============================
    print("Configuring LoRA for parameter-efficient fine-tuning...")
    lora_config = LoraConfig(
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

    print("Applying PEFT to the model...")
    model = get_peft_model(model, lora_config)
    print("Trainable parameters:")
    print_trainable_parameters(model)

    # ==============================
    # 9. Initialize Accelerator with FSDP
    # ==============================
    print("Setting up Accelerator with FullyShardedDataParallelPlugin...")
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
    )

    accelerator = Accelerator(
        mixed_precision="fp16",
        fsdp_plugin=fsdp_plugin
    )

    # ==============================
    # 10. Prepare Datasets
    # ==============================
    print("Tokenizing datasets...")

    def format_prompt(example):
        question = example['body']
        question_type = example['type']
        exact_answer = example['exact_answer']
        ideal_answer = example['ideal_answer']

        if question_type == "yesno":
            prompt = (
                f"### This is an example of yes/no question and the respective answer in the intended format. "
                f"The answer can only use JSON format. The format must be {{\"exact_answer\":\"\", \"ideal_answer\":\"\"}}, "
                f"where exact_answer should be 'yes' or 'no', and ideal_answer is a short conversational response starting "
                f"with yes/no then follow on the explanation.\n### Yes/No question: {question}\n### Answer: {{\"exact_answer\":\"{exact_answer}\", \"ideal_answer\":\"{ideal_answer}\"}}"
            )
        elif question_type == "list":
            prompt = (
                f"### This is an example of list question and the respective answer in the intended format. "
                f"The answer can only use JSON format. The format must be {{\"exact_answer\":[], \"ideal_answer\":\"\"}}, "
                f"where exact_answer is a list of precise key entities to answer the question, and ideal_answer is a short conversational response containing an explanation.\n### List question: {question}\n### Answer: {{\"exact_answer\":{exact_answer}, \"ideal_answer\":\"{ideal_answer}\"}}"
            )
        elif question_type == "factoid":
            prompt = (
                f"### This is an example of factoid question and the respective answer in the intended format. "
                f"The answer can only use JSON format. The format must be {{\"exact_answer\":[], \"ideal_answer\":\"\"}}, "
                f"where exact_answer is a list of precise key entities to answer the question. ideal_answer is a short conversational response containing an explanation.\n### Factoid question: {question}\n### Answer: {{\"exact_answer\":{exact_answer}, \"ideal_answer\":\"{ideal_answer}\"}}"
            )
        elif question_type == "summary":
            prompt = (
                f"### This is an example of summary question and the respective answer in the intended format. "
                f"The reply to the answer should be clear and easy in less than 3 sentences.\n### Summary question: {question}\n### Answer: {ideal_answer}"
            )
        else:
            prompt = (
                f"### Question: {question}\n### Answer: {ideal_answer}"
            )

        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_train_dataset = train_dataset.map(format_prompt, batched=False)
    tokenized_eval_dataset = eval_dataset.map(format_prompt, batched=False)

    # ==============================
    # 11. Data Collator
    # ==============================
    print("Setting up data collator...")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # ==============================
    # 12. Prepare Model and Datasets with Accelerator
    # ==============================
    print("Preparing model and datasets with Accelerator...")
    model, tokenized_train_dataset, tokenized_eval_dataset = accelerator.prepare(
        model, tokenized_train_dataset, tokenized_eval_dataset
    )

    # ==============================
    # 13. Define Training Arguments
    # ==============================
    base_model_name = "mistral7Binstruct"
    run_name = f"{base_model_name}-{project_name}"
    output_dir = run_name

    print("Configuring TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=100,  # Adjust as needed
        learning_rate=2.5e-4,
        logging_steps=10,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        do_eval=True,
        optim="paged_adamw_8bit",
        fp16=True,  # Enable mixed precision
        ddp_find_unused_parameters=False,
    )

    # ==============================
    # 14. Initialize Trainer
    # ==============================
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    # ==============================
    # 14.1 Check parameter placements
    # ==============================

    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device {param.device}")

    # ==============================
    # 15. Start Training
    # ==============================
    print("Starting training...")
    trainer.train()

    # ==============================
    # 16. Save the Fine-Tuned Model
    # ==============================
    print(f"Saving the fine-tuned model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training completed successfully.")

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

if __name__ == "__main__":
    main()