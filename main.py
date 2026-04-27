import pandas as pd
import ollama
import numpy as np
import re

from deep_translator import GoogleTranslator
from langdetect import detect
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
CSV_PATH = "dataset.csv"
FAQ_FILE = "freq_faq.txt"
OLLAMA_MODEL = "phi3"

SUPPORTED_LANGS = ["en", "hi", "mr", "gu"]

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

df["Crop"] = df["Crop"].astype(str).str.lower().str.strip()
df["Fertilizer"] = df["Fertilizer"].astype(str).str.lower().str.strip()

ALL_CROPS = df["Crop"].unique().tolist()

# =========================
# FAQ
# =========================
def load_faq(path):
    faqs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            blocks = f.read().split("\n\n")
        for b in blocks:
            if "Question:" in b and "Answer:" in b:
                q = b.split("Question:")[1].split("Answer:")[0].strip().lower()
                a = b.split("Answer:")[1].strip()
                faqs.append((q, a))
    except:
        pass
    return faqs

faqs = load_faq(FAQ_FILE)
faq_q = [q for q, _ in faqs]
faq_a = [a for _, a in faqs]

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
faq_emb = model.encode(faq_q)

# =========================
# LANGUAGE
# =========================
def detect_lang(text):
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGS else "en"
    except:
        return "en"

def translate(text, src, tgt):
    if src == tgt:
        return text
    try:
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except:
        return text

# =========================
# CLEAN
# =========================
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

# =========================
# NORMALIZE
# =========================
CROP_MAP = {
    "gehu": "wheat",
    "gehun": "wheat",
    "गेहूं": "wheat",
    "chawal": "rice",
    "dhan": "rice",
    "धान": "rice",
    "makka": "maize",
}

def normalize_query(q):
    return " ".join([CROP_MAP.get(w, w) for w in q.lower().split()])

# =========================
# INTENT
# =========================
def detect_intent(q):
    if any(w in q for w in ["fertilizer","खत","ખાતર"]):
        return "fertilizer"

    if any(w in q for w in ["pest","insect","bug","कीड़े","જીવાત"]):
        return "pest"

    if any(w in q for w in ["yojana","scheme","pm kisan","सरकार","योजना"]):
        return "scheme"

    if any(w in q for w in ["water","irrigation","पानी"]):
        return "irrigation"

    return "general"

# =========================
# DATASET
# =========================
def search_dataset(q):
    for crop in ALL_CROPS:
        if crop in q:
            res = df[df["Crop"] == crop]
            if not res.empty:
                ferts = ", ".join(res["Fertilizer"].unique())
                return f"For {crop}, use {ferts}. Follow soil test."
    return None

# =========================
# FAQ
# =========================
def search_faq(q):
    emb = model.encode([q])
    scores = cosine_similarity(emb, faq_emb)[0]
    idx = np.argmax(scores)

    if scores[idx] > 0.5:
        return faq_a[idx]
    return None

# =========================
# LLM
# =========================
def ask_llm(q):
    prompt = f"""
You are helping farmers.

Rules:
- Use very simple language
- Give short practical advice
- No technical words
- 2-3 sentences only

Question: {q}
Answer:
"""
    try:
        res = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        ans = clean_text(res["message"]["content"])
        return ans if ans.endswith(".") else ans + "."
    except:
        return "Please consult your local agriculture officer."

# =========================
# MAIN CHATBOT
# =========================
def chatbot(user_input):

    if len(user_input.strip()) < 3:
        return "Please ask a proper farming question."

    lang = detect_lang(user_input)
    en = normalize_query(translate(user_input, lang, "en"))

    intent = detect_intent(en)

    # 🔥 FIXED IMPORTANT CASES

    # Yellow leaves
    if "yellow" in en:
        ans = "Leaves turning yellow may be due to low nutrients or insects. Use proper fertilizer and check plants."
        return translate(ans, "en", lang)

    # Pest
    if intent == "pest":
        ans = "Check plants regularly, remove damaged leaves and spray neem solution."
        return translate(ans, "en", lang)

    # PM Kisan
    if "pm kisan" in en:
        ans = "Under PM Kisan scheme, farmers get ₹6000 per year, paid in 3 parts every 4 months directly in bank account."
        return translate(ans, "en", lang)

    # Irrigation
    if intent == "irrigation":
        if "rice" in en:
            ans = "Keep 2–5 cm water in field. Do not overfill. Ensure water can drain."
        else:
            ans = "Give water based on crop need. Do not overwater."
        return translate(ans, "en", lang)

    # Fertilizer
    if intent == "fertilizer":
        ans = search_dataset(en)
        if ans:
            return translate(ans, "en", lang)

    # NORMAL FLOW
    ans = search_faq(en)

    if not ans:
        ans = ask_llm(en)

    return translate(ans, "en", lang)