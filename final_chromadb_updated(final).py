from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import torch
import os

from sentence_transformers import SentenceTransformer
import chromadb

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Allow CORS (so frontend can call this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
clip_text_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
clip_image_model = SentenceTransformer("clip-ViT-B-32")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="skin_diseases")

# Embedding utils
def get_text_embedding(text):
    return clip_text_model.encode(text).tolist()

def get_image_embedding(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return clip_image_model.encode(image).tolist()

def combine_embeddings(text_emb, image_emb):
    return [(x + y) / 2 for x, y in zip(text_emb, image_emb)]

# FastAPI models
class TextQuery(BaseModel):
    symptoms: str

@app.get("/")
def root():
    return {"message": "Skin Disease Diagnosis API running"}

@app.post("/search_and_diagnose/")
async def search_and_diagnose(symptoms: str = Form(...), image: UploadFile = File(None)):
    try:
        # Get embeddings
        text_embedding = get_text_embedding(symptoms)
        image_embedding = get_image_embedding(await image.read()) if image else None

        final_embedding = text_embedding
        if image_embedding:
            final_embedding = combine_embeddings(text_embedding, image_embedding)

        # Search ChromaDB
        results = collection.query(final_embedding, n_results=5)
        retrieved_cases = results["metadatas"]

        # Prepare context for diagnosis
        context = "\n".join([
            f"Disease: {case['disease']}\nSymptoms: {case['symptoms']}"
            for case in retrieved_cases
        ])

        # Generate diagnosis using LLM
        prompt = f"""A user has uploaded a skin image and described symptoms.
Medical Cases:\n{context}
User Symptoms: {symptoms}
Provide a possible diagnosis and remedy:"""

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = llm_model.generate(input_ids, max_length=300)
        diagnosis = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "diagnosis": diagnosis,
            "retrieved_cases": retrieved_cases
        }

    except Exception as e:
        return {"error": str(e)}
