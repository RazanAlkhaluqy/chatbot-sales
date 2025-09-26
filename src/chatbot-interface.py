# chatbot-interface.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# Model settings
model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4"
device = "cuda" if torch.cuda.is_available() else "cpu"
offload_folder = "offload_cache"
os.makedirs(offload_folder, exist_ok=True)


st.title("🤖 Chatbot Interface")

# Load tokenizer and model
with st.spinner("Loading model... This may take a while."):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    #     offload_folder=offload_folder
    # )
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float32  # or bfloat16 if GPU
    )
    model.eval()

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)


# Chat interface
user_input = st.text_input("أدخل رسالتك هنا:")

if user_input:
    response = llama_pipeline(user_input)[0]["generated_text"]
    st.write(f"🤖 الروبوت: {response}")
