from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer, GPTNeoForCausalLM
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from bson import ObjectId
from fastapi.encoders import jsonable_encoder
import torch
import numpy as np

client = MongoClient("mongodb://localhost:27017")
db = client['celular_db']
collection = db['offers']

app = FastAPI()

model_sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

model_name_gpt = "EleutherAI/gpt-neo-125M"
tokenizer_gpt = AutoTokenizer.from_pretrained(model_name_gpt)
model_gpt = GPTNeoForCausalLM.from_pretrained(model_name_gpt).to("cpu")

class Offer(BaseModel):
    name: str
    price: float
    link: str
    other_info: str

class OfferList(BaseModel):
    offers: list[Offer]


def generate_sbert_embeddings(text):
    embeddings = model_sbert.encode(text, convert_to_numpy=True)
    print(f"Generated embeddings: {embeddings[:5]}")  
    return embeddings.tolist()


def generate_gpt_neo_description(offer):
    prompt = f"Create a brief description for the product {offer.name}. Additional information: {offer.other_info}."

    inputs = tokenizer_gpt(prompt, return_tensors="pt").to("cpu")
    outputs = model_gpt.generate(
        inputs['input_ids'],
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.5, 
        no_repeat_ngram_size=3 
    )
    
    
    description = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
    
  
    description_cleaned = description.replace(prompt, "").strip()
    
    return description_cleaned



@app.post("/process-offers/")
async def process_offers(offers: OfferList):
    for offer in offers.offers:
        description = generate_gpt_neo_description(offer)
        text_for_embedding = f"{offer.name}, {offer.price}, {offer.other_info}"
        embedding = generate_sbert_embeddings(text_for_embedding)
        
        document = {
            "name": offer.name,
            "price": offer.price,
            "link": offer.link,
            "description": description,
            "embedding": embedding
        }
        
        collection.insert_one(document)

    return {"message": "Offers processed and saved successfully!"}


def cosine_similarity(v1, v2):
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"Calculated similarity: {sim}") 
    return sim


class PromptRequest(BaseModel):
    prompt: str

def format_document(document):
    if "_id" in document:
        document["_id"] = str(document["_id"])
    return document

@app.post("/search-offers/")
async def search_offers(prompt_request: PromptRequest):
    prompt_embedding = generate_sbert_embeddings(prompt_request.prompt)
    
    offers = collection.find()
    
    results = []
    
    for offer in offers:
        offer = format_document(offer)
        
        saved_embedding = np.array(offer['embedding'])
        similarity = cosine_similarity(prompt_embedding, saved_embedding)
        
        results.append({
            "offer": offer,
            "similarity": similarity
        })
    
    if results:
        most_similar_result = max(results, key=lambda x: x['similarity'])
    else:
        return {"message": "No results found"}
    return jsonable_encoder({"most_similar_result": most_similar_result})
