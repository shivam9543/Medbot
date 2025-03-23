import os
import torch
import clip
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import streamlit as st

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Sentence Transformers
text_image_model = SentenceTransformer("clip-ViT-B-32")
text_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="skin_diseases")

# Load Mistral Model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Embedding Functions
def get_text_embedding(text):
    return text_model.encode(text).tolist()

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    return text_image_model.encode(image).tolist()

def search_skin_disease(query_text=None, query_image=None, top_k=5):
    if query_image:
        embedding = get_image_embedding(query_image)
    elif query_text:
        embedding = get_text_embedding(query_text)
    else:
        raise ValueError("Provide either an image or a text description.")
    results = collection.query(embedding, n_results=top_k)
    return results["metadatas"]

def generate_diagnosis(retrieved_cases, user_query):
    context = "\n".join([f"Disease: {case['disease']}\nSymptoms: {case['symptoms']}" for case in retrieved_cases])
    prompt = f"""
A user has uploaded an image and described their symptoms.
Based on the following medical cases, provide a possible diagnosis and remedies.

Medical Cases:
{context}

User Query:
{user_query}

Analyze and suggest possible remedies or recommend a dermatologist if unsure.
Diagnosis:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**inputs, max_length=500)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("🩺 Skin Disease Diagnosis App")
st.write("Upload your skin image and describe symptoms to get AI-powered diagnosis.")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
user_symptoms = st.text_area("Describe your symptoms")

if st.button("Get Diagnosis"):
    if not uploaded_image or not user_symptoms:
        st.warning("Please upload both image and symptoms.")
    else:
        temp_path = f"temp_{uploaded_image.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_image.read())

        st.info("Searching similar cases and generating diagnosis...")
        retrieved = search_skin_disease(query_text=user_symptoms, query_image=temp_path)
        diagnosis = generate_diagnosis(retrieved, user_symptoms)

        st.subheader("Diagnosis & Recommendation")
        st.write(diagnosis)

        os.remove(temp_path)
