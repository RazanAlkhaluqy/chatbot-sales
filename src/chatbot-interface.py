import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# Load model (CPU/GPU)
model_id = "meta-llama/Llama-3.1-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1)

# Streamlit UI
st.title("🤖 LLaMA Chatbot")
user_input = st.text_input("أدخل سؤالك هنا:")

if st.button("إرسال") and user_input:
    response = llama_pipeline(user_input, max_new_tokens=200)[0]["generated_text"]
    st.text_area("الرد:", value=response, height=200)
