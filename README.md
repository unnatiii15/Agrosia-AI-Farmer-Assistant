# 🌾 Agrosia – AI Farmer Assistant

Agrosia is a multilingual AI assistant designed to help farmers with crop-related queries such as fertilizers, pest control, irrigation, and government schemes.

## 🚀 Features

- 🌍 Multilingual support (English, Hindi, Gujarati, Marathi)
- 🎤 Voice input & audio output
- 📊 Dataset-based accurate fertilizer recommendations
- 📚 FAQ-based answers for common problems
- 🤖 AI-powered responses using LLM (Ollama)
- 💬 Clean chat UI

---

## 🧠 How It Works

1. Detects user language  
2. Translates input to English  
3. Uses:
   - Dataset → fertilizers  
   - FAQ → common queries  
   - Rules → irrigation, pests  
   - LLM → general queries  
4. Returns answer in original language  

---

## 🛠️ Tech Stack

- Python (FastAPI)
- Ollama (LLM)
- Sentence Transformers
- HTML, CSS, JavaScript
- Speech Recognition & Synthesis APIs

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload