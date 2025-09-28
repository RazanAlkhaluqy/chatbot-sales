
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import streamlit as st

# st.set_page_config(page_title="🤖 Chatbot Interface", page_icon="🤖")
# st.title("🤖 Chatbot Interface")
# st.markdown("يرجى الانتظار أثناء تحميل النموذج الكبير...")

# # -------------------------
# # Model configuration
# # -------------------------
# # model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4"
# # model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
# # model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4"

# model_id = "meta-llama/Llama-3.1-8B-Instruct"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# st.text(f"النموذج سيعمل على: {device}")

# # -------------------------
# # Load model with spinner
# # -------------------------
# with st.spinner("جارٍ تحميل النموذج، هذا قد يستغرق عدة دقائق..."):
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map=None,  # CPU only
#         torch_dtype=torch.float32
#     )
#     model.eval()

#     llama_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=200,
#         do_sample=True,
#         temperature=0.7,
#         top_k=50,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id
#     )

# st.success("تم تحميل النموذج ✅")

# # -------------------------
# # Session state
# # -------------------------
# if "history" not in st.session_state:
#     st.session_state.history = []

# user_input = st.text_input("أنت:")

# if user_input:
#     with st.spinner("الروبوت يفكر..."):
#         response = llama_pipeline(user_input)[0]["generated_text"]
#     st.session_state.history.append(("أنت", user_input))
#     st.session_state.history.append(("🤖 الروبوت", response))

# # Display chat history
# for sender, msg in st.session_state.history:
#     st.markdown(f"**{sender}:** {msg}")

#########################################################################
# import streamlit as st
# from transformers import pipeline

# st.set_page_config(page_title="🤖 Chatbot Interface", page_icon="🤖")
# st.title("🤖 Chatbot Interface")
# st.markdown("يرجى الانتظار أثناء الاتصال بالنموذج...")

# # -------------------------
# # Model configuration via Hugging Face API
# # -------------------------
# model_id = "meta-llama/Llama-3.1-8B-Instruct"

# # Create a text-generation pipeline using the Hugging Face API
# llama_pipeline = pipeline(
#     "text-generation",
#     model=model_id,        # This will use your logged-in token
#     device_map="auto",     # Use GPU if available
#     max_new_tokens=200,
#     do_sample=True,
#     temperature=0.7,
#     top_k=50,
#     pad_token_id=50256
# )

# # -------------------------
# # Session state
# # -------------------------
# if "history" not in st.session_state:
#     st.session_state.history = []

# user_input = st.text_input("أنت:")

# if user_input:
#     with st.spinner("الروبوت يفكر..."):
#         response = llama_pipeline(user_input)[0]["generated_text"]

#     st.session_state.history.append(("أنت", user_input))
#     st.session_state.history.append(("🤖 الروبوت", response))

# # Display chat history
# for sender, msg in st.session_state.history:
#     st.markdown(f"**{sender}:** {msg}")
#################################### to use API unstead of locally
# from huggingface_hub import InferenceClient
# HF_TOKEN="hf_agav"
# MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
# client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
#########################################

#-------------- correct runing code by API  from hugging face -------------#


# import streamlit as st
# from huggingface_hub import InferenceClient
# import os
# from dotenv import load_dotenv

# # -------------------------
# # Hugging Face API config
# # -------------------------

# # Load .env from parent folder
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# HF_TOKEN = os.getenv("HF_TOKEN")
# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)


# # -------------------------
# # Page config
# # -------------------------
# st.set_page_config(page_title="🤖 Chatbot Interface", page_icon="🤖", layout="centered")

# # Title
# st.markdown("<h1 style='text-align:center; color: #4f008c'>🤖 Smart Assistant</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align:center; color:gray;'>Powered by Hugging Face API</p>", unsafe_allow_html=True)

# # -------------------------
# # Session state
# # -------------------------
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Chat container
# chat_container = st.container()

# # User input at bottom
# with st.form("chat_form", clear_on_submit=True):
#     user_input = st.text_input("💬 Type your message:", "")
#     submitted = st.form_submit_button("Send")

# if submitted and user_input:
#     with st.spinner("🤖 Thinking..."):
#         response = client.chat_completion(
#             model=MODEL_ID,
#             messages=[
#                 {"role": "system", "content": "You are a smart assistant"},
#                 {"role": "user", "content": user_input}
#             ],
#             max_tokens=200,
#             temperature=0.7,
#             top_p=0.9
#         )
#         reply = response.choices[0].message["content"]

#     st.session_state.history.append(("you", user_input))
#     st.session_state.history.append(("🤖 bot", reply))


# # -------------------------
# # Display chat history
# # -------------------------
# with chat_container:
#     for sender, msg in st.session_state.history:
#         if sender == "you":
#             st.markdown(f"""
#                 <div style="background-color: #E6E6FA; padding:10px; border-radius:10px; margin:5px; text-align:right;">
#                 <b>🧑 You:</b> {msg}
#                 </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#                 <div style="background-color:#E6E6E6; padding:10px; border-radius:10px; margin:5px; text-align:left;">
#                 <b>🤖 Assistant:</b> {msg}
#                 </div>
#             """, unsafe_allow_html=True)

#----------------------------------------------------------#

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Model config (local GPT-2)
# -------------------------
MODEL_ID = "gpt2"  # You can also try "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Put model in eval mode (faster, avoids training mode)
model.eval()

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="🤖 Chatbot Interface", page_icon="🤖", layout="centered")

# Title
st.markdown("<h1 style='text-align:center; color:#4f008c'>🤖 Smart Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Running GPT-2 locally</p>", unsafe_allow_html=True)

# -------------------------
# Session state
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Chat container
chat_container = st.container()

# User input at bottom
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Type your message:", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    with st.spinner("🤖 Thinking..."):
        # Prepare input (append conversation history for context)
        conversation = "\n".join([f"You: {msg}" if sender == "you" else f"Assistant: {msg}" 
                                   for sender, msg in st.session_state.history])
        prompt = conversation + f"\nYou: {user_input}\nAssistant:"

        inputs = tokenizer.encode(prompt, return_tensors="pt")

        # Generate response
        output_ids = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        reply = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)

    # Save chat
    st.session_state.history.append(("you", user_input))
    st.session_state.history.append(("🤖 bot", reply))

# -------------------------
# Display chat history
# -------------------------
with chat_container:
    for sender, msg in st.session_state.history:
        if sender == "you":
            st.markdown(f"""
                <div style="background-color:#E6E6FA; padding:10px; border-radius:10px; margin:5px; text-align:right;">
                <b>🧑 You:</b> {msg}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color:#E6E6E6; padding:10px; border-radius:10px; margin:5px; text-align:left;">
                <b>🤖 Assistant:</b> {msg}
                </div>
            """, unsafe_allow_html=True)
