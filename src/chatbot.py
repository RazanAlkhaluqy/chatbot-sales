import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
import gc
from huggingface_hub import login

# 🔑 تسجيل الدخول باستخدام Hugging Face Token (انسخه مكان XXXXXX)
login("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")

# 1. تحديد النموذج
model_id = "meta-llama/Llama-3.1-8B-Instruct"

print(f"\nبدء تحميل نموذج {model_id}. قد يستغرق هذا وقتًا طويلاً ويستهلك الكثير من الذاكرة.")
print("إذا واجهت مشاكل في الذاكرة، ففكر في استخدام 'load_in_8bit' أو 'load_in_4bit'.")

# 2. تحديد الجهاز (GPU أو CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"النموذج سيعمل على: {device}")

try:
    # تحميل Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # تحميل النموذج
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
        use_auth_token=True
    )
    model.eval()

    # تجهيز pipeline
    llama_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=0 if device == "cuda" else -1,  # -1 معناها CPU
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    print(f"✅ تم تحميل نموذج {model_id} بنجاح على {device}.")

except Exception as e:
    print(f"❌ حدث خطأ أثناء تحميل النموذج: {e}")
    print("تأكد من أنك سجلت الدخول إلى Hugging Face وأن لديك وصول للموديل.")
    exit()


# 🔄 دالة لإرجاع الرد من الشات بوت
def chatbot_response(prompt):
    response = llama_pipeline(prompt)[0]["generated_text"]
    return response


# 4. حلقة المحادثة
print("\n🤖 أهلاً بك في روبوت الدردشة الخاص بك! (اكتب 'exit' للخروج)")
while True:
    user_input = input("أنت: ")
    if user_input.lower() == "exit":
        print("🤖 الروبوت: وداعاً!")
        break
    response = chatbot_response(user_input)
    print(f"🤖 الروبوت: {response}")
