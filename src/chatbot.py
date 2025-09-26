import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "meta-llama/Llama-3.1-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"النموذج سيعمل على: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    model.eval()

    llama_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

except Exception as e:
    print(f"❌ حدث خطأ أثناء تحميل النموذج: {e}")
    exit()

def chatbot_response(prompt):
    return llama_pipeline(prompt)[0]["generated_text"]

print("\n🤖 أهلاً بك في روبوت الدردشة! (اكتب 'exit' للخروج)")
while True:
    user_input = input("أنت: ")
    if user_input.lower() == "exit":
        print("🤖 الروبوت: وداعاً!")
        break
    print(f"🤖 الروبوت: {chatbot_response(user_input)}")
