import os
import clip
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Force CPU-only usage
device = "cpu"

# Load CLIP model (for image features)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Sentence Transformer models
text_image_model = SentenceTransformer("clip-ViT-B-32", device=device)
text_model = SentenceTransformer("distiluse-base-multilingual-cased-v2", device=device)

# Load lightweight text generation LLM (FLAN-T5-Base)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(allow_reset=True))
collection = chroma_client.get_or_create_collection(name="skin_diseases")

# Embedding functions
def get_text_embedding(text):
    return text_model.encode(text).tolist()

def get_image_embedding(image):
    image = image.convert("RGB")
    return text_image_model.encode(image).tolist()

# Search function
def search_skin_disease(query_text=None, query_image=None, top_k=5):
    if query_image:
        image_embedding = get_image_embedding(query_image)
        results = collection.query(image_embedding, n_results=top_k)
    elif query_text:
        text_embedding = get_text_embedding(query_text)
        results = collection.query(text_embedding, n_results=top_k)
    else:
        raise ValueError("Provide either text or image for search.")
    return results["metadatas"]

# Diagnosis generation using lightweight LLM
def generate_diagnosis(retrieved_cases, user_query):
    context = "\n".join([
        f"Disease: {case['disease']}\nSymptoms: {case['symptoms']}" for case in retrieved_cases
    ])

    prompt = f"""
    A user has uploaded an image of a skin condition and described their symptoms.
    Based on the following medical records, provide a possible diagnosis.

    Medical Cases:
    {context}

    User Query:
    {user_query}

    Provide a diagnosis and remedies based on similar cases.
    If a specific diagnosis is not possible, recommend seeing a dermatologist.
    Diagnosis and Recommendations:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Streamlit UI ---
st.set_page_config(page_title="Skin Disease Diagnosis", layout="centered")
st.title("🩺 Skin Disease Diagnosis Assistant")

query_text = st.text_input("Describe your symptoms:")
query_image = st.file_uploader("Or upload an image of the skin condition", type=["jpg", "jpeg", "png", "webp"])

if st.button("Search & Diagnose"):
    with st.spinner("Processing..."):
        image_obj = Image.open(query_image) if query_image else None
        retrieved_cases = search_skin_disease(query_text=query_text, query_image=image_obj)
        diagnosis = generate_diagnosis(retrieved_cases, query_text or "No description provided")

        st.subheader("🔍 Retrieved Medical Cases")
        for case in retrieved_cases:
            st.markdown(f"**Disease**: {case['disease']}")
            st.markdown(f"**Symptoms**: {case['symptoms']}")
            st.markdown("---")

        st.subheader("💡 Diagnosis & Recommendations")
        st.markdown(diagnosis)
