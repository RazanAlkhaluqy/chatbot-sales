import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# Load model (CPU/GPU)
model_id = "meta-llama/Llama-3.1-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1)

