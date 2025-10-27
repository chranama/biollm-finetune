#!/usr/bin/env python3

import sys
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from huggingface_hub import login


def main():
    # ==============================
    # 1. Argument Parsing
    # ==============================
    test_data_path = sys.argv[1] 
    results_path = sys.argv[2]
    model_path = sys.argv[3]
    model_name = sys.argv[4]    

    # ==============================
    # 2. Environment Setup
    # ==============================
    assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation."
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
    # 4. Load Dataset
    # ==============================
    print("Loading dataset...")
    test_dataset = load_dataset("json", data_files=test_data_path)['train']

    # ==============================
    # 5. Load Models and Tokenizers with Quantization
    # ==============================
    print(f"Loading model and tokenizer from checkpoints")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # ==============================
    # 6. Format Prompt
    # ==============================
    def build_prompt(example):
        question = example["body"]
        question_type = example["type"]

        if question_type == "yesno":
            prompt = f"Question: {question}\nAnswer (yes or no): "
        elif question_type == "list":
            prompt = f"Question: {question}\nAnswer (list format): "
        elif question_type == "factoid":
            prompt = f"Question: {question}\nAnswer (short fact): "
        elif question_type == "summary":
            prompt = f"Question: {question}\nSummary Answer: "
        else:
            prompt = f"Question: {question}\nAnswer: "
        
        return prompt

    # ==============================
    # 7. Answer Generation
    # ==============================
    def generate_answer(prompt, model, tokenizer, max_new_tokens=200, temperature=0.7, top_p=0.9):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.split("Answer:")[-1].strip()

    # ==============================
    # 8. Executing over model
    # ==============================
    results = []
    for item in test_dataset:
        prompt = build_prompt(item)
        generated_answer = generate_answer(prompt, model, tokenizer)
        results.append({
            "id": item["id"],
            "question": item["body"],
            "question type": item["type"],
            "prompt": prompt,
            "exact_answer": item.get("exact_answer", ""),
            "ideal_answer": item.get("ideal_answer", ""),
            model_name + "_generated_answer": generated_answer
        })

    # ==============================
    # 9. Saving to file
    # ==============================
    with open(results_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Generated answers saved to {results_path}")

if __name__ == "__main__":
    main()
