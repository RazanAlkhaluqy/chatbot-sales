
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







# #------------------------
# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # --------------- handle with excel data file -----------#
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # -------------------------
# # Load cleaned Excel dataset
# # -------------------------
# df = pd.read_excel("../data/amazon_product_reviews_cleaned.xlsx", engine="openpyxl")

# st.write("Available columns:", df.columns.tolist())

# # Create a column to use as knowledge (customize as needed)
# df["knowledge_text"] = (
#     "Product: " + df["product_name"].astype(str) +
#     " | Category: " + df["category"].astype(str) +
#     " | Price Tier: " + df["price_tier"].astype(str) +
#     " | Discount: " + df["discount_level"].astype(str) +
#     " | Rating: " + df["rating"].astype(str) +
#     " | Rating Count: " + df["rating_count"].astype(str) +
#     " | Review: " + df["cleaned_review"].astype(str) +
#     " | Sentiment: " + df["review_sentiment"].astype(str)
# )
# # Vectorize knowledge for retrieval
# vectorizer = TfidfVectorizer(stop_words="english")
# X = vectorizer.fit_transform(df["knowledge_text"].fillna(""))

# def retrieve_context(query, top_k=3):
#     """Find top-k relevant rows from the dataset for a query"""
#     query_vec = vectorizer.transform([query])
#     scores = cosine_similarity(query_vec, X).flatten()
#     top_indices = scores.argsort()[-top_k:][::-1]
#     return df.iloc[top_indices]["knowledge_text"].tolist()
# # ----------------------------------------------------------#

# # -------------------------
# # Model config (local GPT-2)
# # -------------------------
# MODEL_ID = "gpt2"  # You can also try "gpt2-medium", "gpt2-large", "gpt2-xl"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# # Put model in eval mode (faster, avoids training mode)
# model.eval()

# # -------------------------
# # Page config
# # -------------------------
# st.set_page_config(page_title="🤖 Chatbot Interface", page_icon="🤖", layout="centered")

# # Title
# st.markdown("<h1 style='text-align:center; color:#4f008c'>🤖 Smart Assistant</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align:center; color:gray;'>Running GPT-2 locally</p>", unsafe_allow_html=True)

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


# #---------------------------
# if submitted and user_input:
#     with st.spinner("🤖 Thinking..."):
#         # 1. Retrieve relevant context
#         context_chunks = retrieve_context(user_input, top_k=3)
#         context_text = "\n".join(context_chunks)

#         # 2. Build prompt with context
#         prompt = f"""
#         You are a smart assistant that answers based on the following product data:
#         {context_text}

#         Question: {user_input}
#         Assistant:"""

#         inputs = tokenizer.encode(prompt, return_tensors="pt")
#         output_ids = model.generate(
#             inputs,
#             max_length=inputs.shape[1] + 100,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id
#         )
#         reply = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)

#     # Save chat
#     st.session_state.history.append(("you", user_input))
#     st.session_state.history.append(("🤖 bot", reply))



# # -------------------------
# # Display chat history
# # -------------------------
# with chat_container:
#     for sender, msg in st.session_state.history:
#         if sender == "you":
#             st.markdown(f"""
#                 <div style="background-color:#E6E6FA; padding:10px; border-radius:10px; margin:5px; text-align:right;">
#                 <b>🧑 You:</b> {msg}
#                 </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#                 <div style="background-color:#E6E6E6; padding:10px; border-radius:10px; margin:5px; text-align:left;">
#                 <b>🤖 Assistant:</b> {msg}
#                 </div>
#             """, unsafe_allow_html=True)

# # delete ----------------------

# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # -------------------------
# # Excel data handling
# # -------------------------
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load cleaned dataset
# df = pd.read_excel("../data/amazon_product_reviews_cleaned.xlsx", engine="openpyxl")
# st.write("Available columns:", df.columns.tolist())

# # Create knowledge text
# df["knowledge_text"] = (
#     "Product: " + df["product_name"].astype(str) +
#     " | Category: " + df["category"].astype(str) +
#     " | Price Tier: " + df["price_tier"].astype(str) +
#     " | Discount: " + df["discount_level"].astype(str) +
#     " | Rating: " + df["rating"].astype(str) +
#     " | Rating Count: " + df["rating_count"].astype(str) +
#     " | Review: " + df["cleaned_review"].astype(str) +
#     " | Sentiment: " + df["review_sentiment"].astype(str)
# )

# # =========================================
# # Aggregate sentiment and review stats
# # =========================================

# # Average sentiment per product
# product_avg_sentiment = df.groupby('product_name')['sentiment_numerical'].mean().sort_values(ascending=False)

# # Average sentiment per category
# category_avg_sentiment = df.groupby('category')['sentiment_numerical'].mean().sort_values(ascending=False)

# # Sentiment counts and positive ratio per product
# product_sentiment_counts = df.groupby(['product_name', 'review_sentiment']).size().unstack(fill_value=0)
# product_sentiment_counts['total_reviews'] = product_sentiment_counts.sum(axis=1)
# product_sentiment_counts['positive_ratio'] = product_sentiment_counts.get('POSITIVE', 0) / product_sentiment_counts['total_reviews']

# # Vectorize knowledge for retrieval
# vectorizer = TfidfVectorizer(stop_words="english")
# X = vectorizer.fit_transform(df["knowledge_text"].fillna(""))

# def retrieve_context(query, top_k=2):
#     """Return top-k relevant rows from dataset"""
#     query_vec = vectorizer.transform([query])
#     scores = cosine_similarity(query_vec, X).flatten()
#     top_indices = scores.argsort()[-top_k:][::-1]
#     return df.iloc[top_indices]["knowledge_text"].tolist()


# def handle_numeric_query(user_input):
#     """Check for numeric/factual queries and compute answer in Python."""
#     # Example: Highest rating product
#     if "highest rating" in user_input.lower():
#         top_product = df.loc[df['rating'].idxmax()]['product_name']
#         top_rating = df['rating'].max()
#         return f"The product with the highest rating is {top_product} with a rating of {top_rating}."
    
#     # Example: Average sentiment for a product
#     if "average sentiment" in user_input.lower():
#         # Extract product name from input (naive example)
#         for prod in df['product_name'].unique():
#             if prod.lower() in user_input.lower():
#                 avg = product_avg_sentiment.get(prod, None)
#                 if avg is not None:
#                     return f"The average sentiment score for {prod} is {avg:.2f}."
    
#     return None  # Not a numeric/factual query

# # -------------------------
# # Load GPT-2 locally
# # -------------------------
# MODEL_ID = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
# model.eval()

# # -------------------------
# # Streamlit page config
# # -------------------------
# st.set_page_config(page_title="🤖 Chatbot Interface", page_icon="🤖", layout="centered")
# st.markdown("<h1 style='text-align:center; color:#4f008c'>🤖 Smart Assistant</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align:center; color:gray;'>Running GPT-2 locally</p>", unsafe_allow_html=True)
# #--------------------------
# #---------select model -----------
# st.sidebar.title("Select Model")
# model_choice = st.sidebar.radio(
#     "Choose a model for the chatbot:",
#     ("GPT-2 (local)", "Llama 3 (API/ Groq)")
# )
# st.write(f"Using model: {model_choice}")
# # ---------------------------

# from transformers import AutoTokenizer, AutoModelForCausalLM

# if model_choice == "GPT-2 (local)":
#     MODEL_ID = "gpt2"
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#     model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
#     model.eval()

# elif model_choice == "Llama 3 (local or Groq)":
#     # Option A: Local Llama 3
#     # MODEL_ID = "path_to_llama3_weights"
#     # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#     # model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
#     # model.eval()

#     # Option B: Groq API
#     from groq import Groq
#     groq_client = Groq(api_key="YOUR_API_KEY")

# # -------------------------
# # Session state
# # -------------------------
# if "history" not in st.session_state:
#     st.session_state.history = []

# chat_container = st.container()

# # User input
# with st.form("chat_form", clear_on_submit=True):
#     user_input = st.text_input("💬 Type your message:", "")
#     submitted = st.form_submit_button("Send")

# #-------------------------------
# if submitted and user_input:
#     with st.spinner("🤖 Thinking..."):

#         # 1. Check numeric/factual queries first
#         numeric_answer = handle_numeric_query(user_input)
#         if numeric_answer:
#             reply = numeric_answer
#         else:
#             # 2. Retrieve context from Excel
#             context_chunks = retrieve_context(user_input, top_k=2)
#             context_text = "\n".join(context_chunks)

#             # 3. Limit conversation history
#             conversation = "\n".join([
#                 f"You: {msg}" if sender == "you" else f"Assistant: {msg}"
#                 for sender, msg in st.session_state.history[-4:]
#             ])

#             # 4. Build prompt
#             prompt = f"{context_text}\n{conversation}\nYou: {user_input}\nAssistant:"

#             # 5. Encode and truncate
#             inputs = tokenizer.encode(prompt, return_tensors="pt")
#             max_input_tokens = 900
#             if inputs.shape[1] > max_input_tokens:
#                 inputs = inputs[:, -max_input_tokens:]

#             # 6. Generate response
#             output_ids = model.generate(
#                 inputs,
#                 max_length=inputs.shape[1] + 80,
#                 temperature=0.7,
#                 top_p=0.9,
#                 do_sample=True,
#                 pad_token_id=tokenizer.eos_token_id
#             )
#             reply = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)

#         # Save chat
#         st.session_state.history.append(("you", user_input))
#         st.session_state.history.append(("🤖 bot", reply))






# # -------------------------
# # Display chat history
# # -------------------------
# with chat_container:
#     for sender, msg in st.session_state.history:
#         if sender == "you":
#             st.markdown(f"""
#                 <div style="background-color:#E6E6FA; padding:10px; border-radius:10px; margin:5px; text-align:right;">
#                 <b>🧑 You:</b> {msg}
#                 </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#                 <div style="background-color:#E6E6E6; padding:10px; border-radius:10px; margin:5px; text-align:left;">
#                 <b>🤖 Assistant:</b> {msg}
#                 </div>
#             """, unsafe_allow_html=True)




# #######  Groq
# from groq import Groq

# # Initialize Groq client
# client = Groq(api_key='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 ')

# # Define your message
# messages = [
#     {"role": "user", "content": "Tell me about the latest smartphone."}
# ]

# # Send the message and get the response
# response = client.chat.completions.create(
#     messages=messages,
#     model="llama3-70b-8192"
# )

# print(response.choices[0].message.content)


import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
from dotenv import load_dotenv

# Load .env from parent folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# -------------------------
# Load cleaned Excel dataset
# -------------------------
df = pd.read_excel("../data/amazon_product_reviews_cleaned.xlsx", engine="openpyxl")
# st.write("Available columns:", df.columns.tolist())

# Create knowledge text
df["knowledge_text"] = (
    "Product: " + df["product_name"].astype(str) +
    " | Category: " + df["category"].astype(str) +
    " | Price Tier: " + df["price_tier"].astype(str) +
    " | Discount: " + df["discount_level"].astype(str) +
    " | Rating: " + df["rating"].astype(str) +
    " | Rating Count: " + df["rating_count"].astype(str) +
    " | Review: " + df["cleaned_review"].astype(str) +
    " | Sentiment: " + df["review_sentiment"].astype(str)
)

# -------------------------
# Aggregate sentiment stats
# -------------------------
product_avg_sentiment = df.groupby('product_name')['sentiment_numerical'].mean().sort_values(ascending=False)
category_avg_sentiment = df.groupby('category')['sentiment_numerical'].mean().sort_values(ascending=False)
product_sentiment_counts = df.groupby(['product_name', 'review_sentiment']).size().unstack(fill_value=0)
product_sentiment_counts['total_reviews'] = product_sentiment_counts.sum(axis=1)
product_sentiment_counts['positive_ratio'] = product_sentiment_counts.get('POSITIVE', 0) / product_sentiment_counts['total_reviews']

# -------------------------
# Vectorize knowledge for retrieval
# -------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["knowledge_text"].fillna(""))

def retrieve_context(query, top_k=2):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]["knowledge_text"].tolist()

# -------------------------
# Handle numeric/factual queries
# -------------------------
def handle_numeric_query(user_input):
    if "highest rating" in user_input.lower():
        top_product = df.loc[df['rating'].idxmax()]['product_name']
        top_rating = df['rating'].max()
        return f"The product with the highest rating is {top_product} with a rating of {top_rating}."
    if "lowest rating" in user_input.lower():
        low_product = df.loc[df['rating'].idxmin()]['product_name']
        low_rating = df['rating'].min()
        return f"The product with the lowest rating is {low_product} with a rating of {low_rating}."
    if "average sentiment" in user_input.lower():
        for prod in df['product_name'].unique():
            if prod.lower() in user_input.lower():
                avg = product_avg_sentiment.get(prod, None)
                if avg is not None:
                    return f"The average sentiment score for {prod} is {avg:.2f}."
    if "average sentiment per category" in user_input.lower():
        res = "\n".join([f"{cat}: {score:.2f}" for cat, score in category_avg_sentiment.items()])
        return f"Average sentiment per category:\n{res}"
    return None

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="🤖 Chatbot Interface", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align:center; color:#4f008c'>🤖 Smart Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>RAG + Multi-Model Chatbot</p>", unsafe_allow_html=True)

# -------------------------
# Sidebar model selection
# -------------------------
st.sidebar.title("Select Model")
model_choice = st.sidebar.radio(
    "Choose a model for the chatbot:",
    ("GPT-2 (local)", "Llama 3 (API/ Groq)")
)
st.write(f"Using model: {model_choice}")

# -------------------------
# Load models conditionally
# -------------------------
if model_choice == "GPT-2 (local)":
    MODEL_ID = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

elif model_choice == "Llama 3 (API/ Groq)":
    from groq import Groq
groq_client = Groq(api_key=TOKEN_Groq)
try:
    info = groq_client.models.list()
    print("Connection OK. Available models:", info)
except Exception as e:
    print("API Connection failed:", e)


# -------------------------
# Session state
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []
chat_container = st.container()

# -------------------------
# User input form
# -------------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Type your message:", "")
    submitted = st.form_submit_button("Send")

# -------------------------
# Process user input
# -------------------------
if submitted and user_input:
    with st.spinner("🤖 Thinking..."):

        # Check numeric/factual queries
        numeric_answer = handle_numeric_query(user_input)
        if numeric_answer:
            reply = numeric_answer
        else:
            # Retrieve context
            context_chunks = retrieve_context(user_input, top_k=2)
            context_text = "\n".join(context_chunks)

            # Limit history
            conversation = "\n".join([
                f"You: {msg}" if sender == "you" else f"Assistant: {msg}"
                for sender, msg in st.session_state.history[-4:]
            ])

            prompt = f"{context_text}\n{conversation}\nYou: {user_input}\nAssistant:"

            if model_choice == "GPT-2 (local)":
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                max_input_tokens = 900
                if inputs.shape[1] > max_input_tokens:
                    inputs = inputs[:, -max_input_tokens:]
                output_ids = model.generate(
                    inputs,
                    max_length=inputs.shape[1]+80,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                reply = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)

            elif model_choice == "Llama 3 (API/ Groq)":
                messages = [{"role": "user", "content": user_input}]
                response = groq_client.chat.completions.create(
                    messages=messages,
                    model="llama-3.3-70b-versatile"
                )
                reply = response.choices[0].message.content

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
