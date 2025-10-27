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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,8"  # Use Tesla M10 GPUs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Optimize memory
    assert torch.cuda.is_available(), "CUDA is not available. Please check your installation."

    torch.cuda.empty_cache()

    # Select the main device (Force to cuda:0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    # 5. Load Model and Tokenizer
    # ==============================
    print("Loading model and tokenizer from checkpoint")
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
        device_map=None  # ❌ Disable auto-splitting
    )

    model.eval()  # Set model to evaluation mode

    # 🔥 Move all model weights & LoRA layers to the same GPU
    model.to(device)
    for name, param in model.named_parameters():
        param.to(device)  # Ensure everything is moved correctly
        print(f"✅ {name} is now on {param.device}")

    # ==============================
    # 6. Format Prompt
    # ==============================
    def build_prompt(question, question_type):
        if not isinstance(question, str):
            raise ValueError(f"Question must be a string, got {type(question)}: {question}")

        if question_type == "yesno":
            prompt = (
                f"### This is an example of yes/no question and the respective answer in the intended format. "
                f"The answer can only use JSON format. The format must be {{\"exact_answer\":\"\", \"ideal_answer\":\"\"}}, "
                f"where exact_answer should be 'yes' or 'no', and ideal_answer is a short conversational response starting "
                f"with yes/no then follow on the explanation.\n### An example of a Yes/No question: Is the protein Papilin secreted?\n"
                f"### An example of a Yes/No answer: {{\"exact_answer\":\"yes\", \"ideal_answer\":\"Yes,  papilin is a secreted protein\"}}"
                f"\n### Yes/No question: {question}\n### Answer: "
            )
        elif question_type == "list":
            prompt = (
                f"### This is an example of list question and the respective answer in the intended format. "
                f"The answer can only use JSON format. The format must be {{\"exact_answer\":[], \"ideal_answer\":\"\"}}, "
                f"where exact_answer is a list of precise key entities to answer the question, and ideal_answer is a short conversational response containing an explanation."
                f"\n### List question: {question}\n### Answer: "
            )
        elif question_type == "factoid":
            prompt = (
                f"### This is an example of factoid question and the respective answer in the intended format. "
                f"The answer can only use JSON format. The format must be {{\"exact_answer\":[], \"ideal_answer\":\"\"}}, "
                f"where exact_answer is a list of precise key entities to answer the question. ideal_answer is a short conversational response containing an explanation."
                f"\n### Factoid question: {question}\n### Answer: "
            )
        elif question_type == "summary":
            prompt = (
                f"### Summary question: {question}\n### Answer: "
            )
        else:
            raise ValueError(f"Unknown question type: {question_type}")
        
        return prompt

    # ==============================
    # 7. Answer Generation (Fixed for GPU Placement)
    # ==============================
    def generate_answer(prompt, model, tokenizer, max_new_tokens=100, temperature=0.7, top_p=0.9):
        """
        Generate an answer using the causal language model.
        """
        if not isinstance(prompt, str):
            raise ValueError(f"Expected prompt to be a string, but got {type(prompt)}: {prompt}")

        # Move input tensors to the same device as the model
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Validate input placements
        assert input_ids.device == device, "❌ input_ids not on model device!"
        assert attention_mask.device == device, "❌ attention_mask not on model device!"

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id  # ✅ Fix attention mask issue
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # ==============================
    # 8. Executing Over Model
    # ==============================
    results = []
    for item in test_dataset:
        prompt = build_prompt(item['body'], item['type'])
        print(f"Processing prompt: {prompt}")

        with torch.no_grad():
            generated_answer = generate_answer(prompt, model, tokenizer)

        # Free GPU memory after each iteration
        torch.cuda.empty_cache()

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
    # 9. Saving to File
    # ==============================
    with open(results_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"✅ Generated answers saved to {results_path}")

if __name__ == "__main__":
    main()