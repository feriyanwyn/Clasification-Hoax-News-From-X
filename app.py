import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import torch
import joblib
import os
import requests
import openai
from urllib.parse import urlparse
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import AutoTokenizer, AutoModel

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()

class HoaxPreprocessor:
    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().get_stop_words()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # Case folding
        text = text.lower()
        # Remove URLs, punctuations, and numbers
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization & Stopword Removal
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stopword_remover]
        
        # Stemming
        cleaned_text = ' '.join(tokens)
        return self.stemmer.stem(cleaned_text)

# Cache models and preprocessor so they are only loaded once
@st.cache_resource
def load_ml_models():
    models = {}
    
    # Check if models exist and load them
    if os.path.exists('svm_model.pkl'):
        models["SVM (Linear)"] = joblib.load('svm_model.pkl')
    if os.path.exists('random_forest_model.pkl'):
        models["Random Forest"] = joblib.load('random_forest_model.pkl')
    if os.path.exists('naive_bayes_model.pkl'):
        models["Naive Bayes"] = joblib.load('naive_bayes_model.pkl')
        
    return models

@st.cache_resource
def load_bert_model():
    model_name = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_resource
def load_preprocessor():
    return HoaxPreprocessor()

# --- OpenAI and Domain Metrics Functions ---
def predict_hoax_openai(text: str, api_key: str) -> str:
    """Predicts if a given text is a hoax using OpenAI's GPT model and requests relevant links."""
    if not api_key:
        return "error: OpenAI API key is not set."
    
    openai.api_key = api_key

    prompt_template = f"""Anda adalah asisten deteksi hoaks. Berikan klasifikasi apakah teks berita berikut adalah 'Hoaks' atau 'Fakta'. Jika 'Fakta', berikan juga setidaknya dua tautan artikel berita atau sumber yang relevan dan terpercaya untuk mendukung klaim tersebut. Jika 'Hoaks', berikan juga penjelasan singkat mengapa itu hoaks. Format jawaban Anda seperti ini:

Klasifikasi: [Hoaks/Fakta]
Penjelasan: [Penjelasan singkat]
Tautan Pendukung: [Daftar tautan, jika ada]

Berikut teks beritanya:
\"\"\"{text}\"\"\""""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and accurate hoax detection assistant."},
                {"role": "user", "content": prompt_template.format(text=text)}
            ],
            temperature=0.2, 
            max_tokens=500 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

DOMAIN_METRICS_API_BASE = "https://bot11-q8bt.onrender.com/check-metrics?domain="

def extract_domains_from_text(text: str) -> list:
    """Extracts unique domains from a given text."""
    urls = re.findall(r'https?://(?:www\.)?([^/\s]+)', text)
    domains = set()
    for url in urls:
        parsed_url = urlparse(f"https://{url}")
        domain = parsed_url.netloc
        if domain:
            if domain.startswith('www.'):
                domain = domain[4:]
            domains.add(domain)
    return list(domains)

def get_domain_metrics(domain: str) -> dict:
    """Fetches domain metrics from the external API."""
    api_url = f"{DOMAIN_METRICS_API_BASE}{domain}"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"domain": domain, "error": str(e)}

def get_domain_metrics_from_openai_result(openai_result_text: str) -> dict:
    """Extracts domains from OpenAI result and fetches their metrics."""
    links_section_match = re.search(r'Tautan Pendukung:\s*(.*)', openai_result_text, re.DOTALL)

    if not links_section_match:
        return {"message": "No 'Tautan Pendukung' section found in OpenAI result."}

    links_section = links_section_match.group(1)
    domains = extract_domains_from_text(links_section)

    if not domains:
        return {"message": "No domains found in 'Tautan Pendukung' section."}

    all_metrics = {}
    for domain in domains:
        metrics = get_domain_metrics(domain)
        all_metrics[domain] = metrics

    return all_metrics

def summarize_credibility(domain_metrics: dict) -> str:
    """Summarizes the credibility of news sources based on domain metrics."""
    summary_parts = []
    if not domain_metrics or "message" in domain_metrics:
        return "Tidak ada tautan pendukung atau metrik domain yang ditemukan untuk disimpulkan."

    for domain, metrics in domain_metrics.items():
        if "error" in metrics:
            summary_parts.append(f"**Domain '{domain}'**: Gagal mendapatkan metrik - {metrics['error']}")
            continue

        da = metrics.get("Domain Authority")
        dr = metrics.get("Domain Rating")
        spam_score_str = metrics.get("Spam Score")

        credibility = []

        if da is not None:
            if da >= 70:
                credibility.append("Otoritas Domain (DA) sangat tinggi.")
            elif da >= 40:
                credibility.append("Otoritas Domain (DA) tinggi.")
            else:
                credibility.append("Otoritas Domain (DA) sedang/rendah.")

        if dr is not None:
            if dr >= 70:
                credibility.append("Peringkat Domain (DR) sangat tinggi.")
            elif dr >= 40:
                credibility.append("Peringkat Domain (DR) tinggi.")
            else:
                credibility.append("Peringkat Domain (DR) sedang/rendah.")

        if spam_score_str:
            try:
                spam_score = int(spam_score_str.replace('%', ''))
                if spam_score <= 10:
                    credibility.append("Skor Spam rendah (indikasi kepercayaan).")
                elif spam_score <= 30:
                    credibility.append("Skor Spam sedang (perlu perhatian).")
                else:
                    credibility.append("Skor Spam tinggi (potensi tidak terpercaya).")
            except ValueError:
                credibility.append(f"Skor Spam tidak valid '{spam_score_str}'.")

        if credibility:
            summary_parts.append(f"**Domain '{domain}'**: {' '.join(credibility)}")
        else:
            summary_parts.append(f"**Domain '{domain}'**: Tidak ada metrik yang cukup untuk disimpulkan.")

    return "\n\n".join(summary_parts)

# Main App Layout
st.set_page_config(page_title="Deteksi Hoaks", page_icon="🕵️", layout="wide")

st.title("🕵️ Aplikasi Deteksi Hoaks Berita")
st.markdown("""
Aplikasi ini menggunakan model *Machine Learning* yang telah dilatih pada artikel berita dan hoaks 
(menggunakan IndoBERT untuk ekstraksi fitur) untuk memprediksi apakah suatu berita termasuk **Fakta** atau **Hoaks**.
Selain itu, Anda dapat menggunakan integrasi OpenAI GPT untuk analisis lebih mendalam dan pemeriksaan kredibilitas sumber.
""")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Pengaturan")
    use_openai = st.checkbox("Gunakan Pemeriksaan OpenAI (Advanced)")
    openai_api_key = ""
    if use_openai:
        openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if not openai_api_key:
            st.warning("Masukkan API Key OpenAI untuk menggunakan fitur ini.")

# Load resources
with st.spinner("Memuat model ML lokal (mungkin memakan waktu saat pertama kali dijalankan)..."):
    models = load_ml_models()
    tokenizer, bert_model, device = load_bert_model()
    preprocessor = load_preprocessor()

if not models:
    st.error("Model tidak ditemukan! Pastikan file `svm_model.pkl`, `random_forest_model.pkl`, atau `naive_bayes_model.pkl` berada di direktori yang sama dengan aplikasi.")
    st.stop()

st.subheader("Masukkan Teks Berita")

title = st.text_input("Judul Berita", placeholder="Masukkan judul berita di sini...")
narrative = st.text_area("Narasi / Isi Berita", placeholder="Masukkan teks narasi berita di sini...", height=200)

def predict_hoax_local(text: str) -> dict:
    # 1. Clean the text
    clean_text_input = preprocessor.clean_text(text)
    
    # 2. Get embeddings
    inputs = tokenizer([clean_text_input], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        
    # Mean Pooling
    mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    embedding = (sum_embeddings / sum_mask).cpu().numpy()
    
    # 3. Predict
    predictions = {}
    for name, ml_model in models.items():
        try:
            pred = ml_model.predict(embedding)[0]
            # Based on dataset label mapping: 0 = Hoax, 1 = Fact
            predictions[name] = "Fakta" if pred == 1 else "Hoaks"
        except Exception as e:
            predictions[name] = f"Error: {str(e)}"
            
    return predictions

if st.button("Cek Teks", type="primary"):
    if not title and not narrative:
        st.warning("Silakan masukkan judul atau narasi berita terlebih dahulu.")
    else:
        combined_text = f"{title} {narrative}".strip()
        
        st.markdown("### Hasil Prediksi")
        
        # Local ML Models
        with st.spinner("Menganalisis teks dengan Model ML Lokal..."):
            predictions = predict_hoax_local(combined_text)
            
        st.subheader("🤖 Model Machine Learning (Lokal)")
        cols = st.columns(len(predictions))
        for col, (model_name, pred) in zip(cols, predictions.items()):
            col.markdown(f"**{model_name}**")
            if pred == "Hoaks":
                col.error("🚨 Hoaks")
            elif pred == "Fakta":
                col.success("✅ Fakta")
            else:
                col.warning(pred)
                
        # OpenAI Check
        if use_openai:
            if not openai_api_key:
                st.error("API Key OpenAI belum dimasukkan. Lewati pemeriksaan OpenAI.")
            else:
                st.markdown("---")
                st.subheader("🧠 Analisis Mendalam OpenAI (GPT)")
                with st.spinner("Menghubungi OpenAI..."):
                    openai_result = predict_hoax_openai(combined_text, openai_api_key)
                    
                st.info(openai_result)
                
                with st.spinner("Mengambil Metrik Kredibilitas Domain..."):
                    domain_metrics = get_domain_metrics_from_openai_result(openai_result)
                    
                st.subheader("📊 Ringkasan Kredibilitas Berita")
                credibility_summary = summarize_credibility(domain_metrics)
                st.write(credibility_summary)
                
                with st.expander("Lihat Detail Metrik Domain"):
                    st.json(domain_metrics)

st.markdown("---")
st.caption("Proyek Deteksi Hoaks menggunakan IndoBERT + SVM / Random Forest / Naive Bayes & OpenAI GPT.")
