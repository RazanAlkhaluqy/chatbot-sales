
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import streamlit as st

st.set_page_config(page_title="🤖 Chatbot Interface", page_icon="🤖")
st.title("🤖 Chatbot Interface")
st.markdown("يرجى الانتظار أثناء تحميل النموذج الكبير...")

# -------------------------
# Model configuration
# -------------------------
# model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4"
# model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
# model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4"

model_id = "meta-llama/Llama-3.1-8B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
st.text(f"النموذج سيعمل على: {device}")

# -------------------------
# Load model with spinner
# -------------------------
with st.spinner("جارٍ تحميل النموذج، هذا قد يستغرق عدة دقائق..."):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=None,  # CPU only
        torch_dtype=torch.float32
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

st.success("تم تحميل النموذج ✅")

# -------------------------
# Session state
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("أنت:")

if user_input:
    with st.spinner("الروبوت يفكر..."):
        response = llama_pipeline(user_input)[0]["generated_text"]
    st.session_state.history.append(("أنت", user_input))
    st.session_state.history.append(("🤖 الروبوت", response))

# Display chat history
for sender, msg in st.session_state.history:
    st.markdown(f"**{sender}:** {msg}")
