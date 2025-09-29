# # -------------------------
# # Model configuration
# # -------------------------
# # model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4"
# # model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
# # model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4"

# model_id = "meta-llama/Llama-3.1-8B-Instruct"



import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
from dotenv import load_dotenv

# Load .env from parent folder
#dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv()

TOKEN_Groq = os.getenv("TOKEN_Groq")
if not TOKEN_Groq:
    raise ValueError("TOKEN_Groq is not set. Check your .env file path and contents.")


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

# def retrieve_context(query, top_k=2):
#     query_vec = vectorizer.transform([query])
#     scores = cosine_similarity(query_vec, X).flatten()
#     top_indices = scores.argsort()[-top_k:][::-1]
#     return df.iloc[top_indices]["knowledge_text"].tolist()

def retrieve_context(query, top_k=2):
    df = st.session_state.df  # always use the active dataset

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, vectorizer.transform(df["knowledge_text"])).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
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
#--------------------------
#-------------------------
# Chatbot with SQL + Retrieval
#-------------------------
import re

def handle_sql_query(user_input):
    try:
        # Simple heuristic: if user asks about "average", "max", "min", "count"
        if re.search(r"\b(average|max|min|count|sum|total)\b", user_input.lower()):
            # Example: user asks "average rating"
            if "rating" in user_input.lower():
                query = "SELECT AVG(rating) as avg_rating FROM uploaded_data;"
                result = pd.read_sql(query, conn).iloc[0,0]
                return f"The average rating is {result:.2f}."
            
            # Example: "how many rows?"
            if "rows" in user_input.lower():
                query = "SELECT COUNT(*) as row_count FROM uploaded_data;"
                result = pd.read_sql(query, conn).iloc[0,0]
                return f"The dataset contains {result} rows."

        return None
    except Exception as e:
        return f"⚠️ Could not process SQL query: {e}"


#--------------------------
# -------------------------
# File uploader for Excel
# -------------------------
# uploaded_file = st.sidebar.file_uploader("📂 Upload your Excel file", type=["xlsx", "csv"])

# if uploaded_file:
#     # Save to a temporary location
#     import tempfile
#     temp_dir = tempfile.mkdtemp()
#     file_path = os.path.join(temp_dir, uploaded_file.name)
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Load into DataFrame
#     if file_path.endswith(".csv"):
#         df = pd.read_csv(file_path)
#     else:
#         df = pd.read_excel(file_path, engine="openpyxl")

#     st.success(f"✅ File uploaded successfully! Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# -------------------------
# File upload handler
# -------------------------
uploaded_file = st.file_uploader("📂 Upload an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # ✅ Ensure 'knowledge_text' exists
    if "knowledge_text" not in df.columns:
        df["knowledge_text"] = df.astype(str).apply(lambda row: " | ".join(row.values), axis=1)

    st.session_state.df = df  # store uploaded dataframe
    st.success("✅ File uploaded and ready! Using this as chatbot knowledge.")
else:
    # fallback: load your default cleaned dataset
    df = pd.read_excel("../data/amazon_product_reviews_cleaned.xlsx")

    # ✅ Ensure 'knowledge_text' exists for default dataset too
    if "knowledge_text" not in df.columns:
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

    st.session_state.df = df
    st.info("ℹ️ No file uploaded. Using default Amazon reviews dataset.")


#-------------------------

#-------------------------
# Convert to  SQLite
#-------------------------

import sqlite3

# Create SQLite DB in memory (or on disk)
conn = sqlite3.connect(":memory:")  # ":memory:" keeps it fast and temporary
df.to_sql("uploaded_data", conn, if_exists="replace", index=False)

# # Test query
# test_query = pd.read_sql("SELECT * FROM uploaded_data LIMIT 5;", conn)
# st.write("Preview of uploaded data:", test_query)
#-------------------------
#-------------------------


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

# if submitted and user_input:
#     with st.spinner("🤖 Thinking..."):

#         # Check numeric/factual queries
#         numeric_answer = handle_numeric_query(user_input)
#         if numeric_answer:
#             reply = numeric_answer
#         else:
#             # Retrieve context
#             context_chunks = retrieve_context(user_input, top_k=2)
#             context_text = "\n".join(context_chunks)

#             # Limit history
#             conversation = "\n".join([
#                 f"You: {msg}" if sender == "you" else f"Assistant: {msg}"
#                 for sender, msg in st.session_state.history[-4:]
#             ])

#             prompt = f"{context_text}\n{conversation}\nYou: {user_input}\nAssistant:"

#             if model_choice == "GPT-2 (local)":
#                 inputs = tokenizer.encode(prompt, return_tensors="pt")
#                 max_input_tokens = 900
#                 if inputs.shape[1] > max_input_tokens:
#                     inputs = inputs[:, -max_input_tokens:]
#                 output_ids = model.generate(
#                     inputs,
#                     max_length=inputs.shape[1]+80,
#                     temperature=0.7,
#                     top_p=0.9,
#                     do_sample=True,
#                     pad_token_id=tokenizer.eos_token_id
#                 )
#                 reply = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)

#             elif model_choice == "Llama 3 (API/ Groq)":
#                 messages = [{"role": "user", "content": user_input}]
#                 response = groq_client.chat.completions.create(
#                     messages=messages,
#                     model="llama-3.3-70b-versatile"
#                 )
#                 reply = response.choices[0].message.content

#         # Save chat
#         st.session_state.history.append(("you", user_input))
#         st.session_state.history.append(("🤖 bot", reply))
#############################
if submitted and user_input:
    with st.spinner("🤖 Thinking..."):

        # Step A: Try SQL query handling
        sql_answer = handle_sql_query(user_input) if 'conn' in globals() else None
        if sql_answer:
            reply = sql_answer

        else:
            # Step B: Check numeric/factual queries (your old logic)
            numeric_answer = handle_numeric_query(user_input)
            if numeric_answer:
                reply = numeric_answer

            else:
                # Step C: Retrieve context with TF-IDF
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
                        max_length=inputs.shape[1] + 80,
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

############################


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
