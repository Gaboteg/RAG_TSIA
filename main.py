import os
import numpy as np
import pdfplumber
import faiss
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import re

# Configuración de los modelos
model_name = "meta-llama/LLaMA-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="mitoken")
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token="mitoken")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Rutas para archivos
embeddings_path = './vector_store_faiss.index'
sentences_path = './vector_store_sentences.pkl'
pdf_path = './PDF-Books/LIBRO IFSSA Anatomia.y.Fisiologia.Humana.Marieb.pdf'

# Función para extraer texto y dividirlo en párrafos
def extract_paragraphs_from_pdf(pdf_path):
    paragraphs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Dividir el texto de la página en párrafos usando saltos de línea dobles
                paragraphs.extend(text.split('\n\n'))
    # Filtrar cualquier elemento vacío en caso de que existan líneas en blanco
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


# Crear o cargar vector store
def load_or_create_vector_store():
    if os.path.exists(embeddings_path) and os.path.exists(sentences_path):
        # Cargar FAISS y fragmentos
        index = faiss.read_index(embeddings_path)
        with open(sentences_path, 'rb') as f:
            sentences = pickle.load(f)
        #print("Vector store cargado desde archivo.")
    else:
        # Extraer texto y dividir en oraciones
        sentences = extract_paragraphs_from_pdf(pdf_path)
        embeddings = embedding_model.encode(sentences)

        # Crear FAISS index y almacenar embeddings
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(np.array(embeddings, dtype=np.float32))
        faiss.write_index(index, embeddings_path)
        
        # Guardar oraciones
        with open(sentences_path, 'wb') as f:
            pickle.dump(sentences, f)
        #print("Vector store creado y guardado en archivo.")
    
    return sentences, index

# Obtener embedding de la pregunta
def get_question_embedding(question):
    return embedding_model.encode([question])[0]

# Encontrar fragmentos más relevantes usando FAISS
def find_most_relevant_fragments(question, sentences, index, top_k=3):
    question_embedding = get_question_embedding(question).astype(np.float32)
    _, top_indices = index.search(np.array([question_embedding]), top_k)
    return [sentences[i] for i in top_indices[0]]

def generate_response_with_context(question, context):
    prompt = (
        f"Contexto: {context}\n\n"
        f"Pregunta: {question}\n"
        "Por favor, proporciona una respuesta detallada y precisa basada en el contexto. Si la respuesta no se encuentra en el contexto respondeme de forma cordial que no se de ese tema. No respondas de tus conocimientos previos, ni te inventes una respuesta\nRespuesta:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7#,
            #top_p=0.9,
            #top_k=50
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Buscar el índice donde comienza "Respuesta:"
    respuesta_index = response.find("Respuesta:")  # Encuentra la posición de "Respuesta:"
    
    if respuesta_index != -1:  # Si se encontró "Respuesta:"
        return response[respuesta_index + len("Respuesta:"):].strip()  # Retorna el texto después de "Respuesta:"
    else:
        return response.strip()  # Si no se encontró, devuelve la respuesta completa




# Cargar o crear vector store
sentences, index = load_or_create_vector_store()

# Ejemplo de consulta
question = "¿Quien fue Rockefeller?"
relevant_fragments = find_most_relevant_fragments(question, sentences, index, top_k=3)
context = " ".join(relevant_fragments)

# Generar respuesta
response = generate_response_with_context(question, context)
"""
print("Fragmentos relevantes:")
for fragment in relevant_fragments:
    print(f"Fragmento: {fragment[:50]}...")  # Muestra los primeros 50 caracteres
"""
print("Pregunta: ", question)
print("Respuesta generada:", response)
