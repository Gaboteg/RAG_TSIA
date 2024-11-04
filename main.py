import os
import numpy as np
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import pickle

# Cargar el modelo de lenguaje y el tokenizador de LLaMA
model_name = "meta-llama/LLaMA-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="mitoken")
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token="mitoken")

# Cargar el modelo de embeddings (SentenceTransformer)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Rutas para los archivos de vector store
embeddings_path = './vector_store_embeddings.npy'
fragments_path = './vector_store_fragments.pkl'
pdf_path = './PDF-Books/LIBRO IFSSA Anatomia.y.Fisiologia.Humana.Marieb.pdf'

# Función para extraer texto de un archivo PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Asegúrate de que no sea None
    return text

# Cargar o crear vector store
def load_or_create_vector_store():
    if os.path.exists(embeddings_path) and os.path.exists(fragments_path):
        # Cargar embeddings y fragmentos desde los archivos
        fragment_embeddings = np.load(embeddings_path)
        with open(fragments_path, 'rb') as f:
            text_fragments = pickle.load(f)
        print("Vector store cargado desde archivo.")
    else:
        # Extraer texto y dividir en fragmentos
        pdf_text = extract_text_from_pdf(pdf_path)
        text_fragments = pdf_text.split('\n\n')  # Ajusta el delimitador según el contenido del PDF

        # Crear vector store calculando embeddings
        fragment_embeddings = embedding_model.encode(text_fragments)

        # Guardar embeddings y fragmentos
        np.save(embeddings_path, fragment_embeddings)
        with open(fragments_path, 'wb') as f:
            pickle.dump(text_fragments, f)
        print("Vector store creado y guardado en archivo.")
    
    return text_fragments, fragment_embeddings

# Función para obtener el embedding de una pregunta
def get_question_embedding(question):
    return embedding_model.encode([question])[0]

# Función para encontrar los fragmentos más relevantes
def find_most_relevant_fragments(question, text_fragments, fragment_embeddings, top_k=3):
    question_embedding = get_question_embedding(question)

    # Calcular similitud usando distancia coseno
    similarities = cosine_similarity([question_embedding], fragment_embeddings)
    
    # Obtener los índices de los fragmentos más similares
    most_similar_indices = np.argsort(similarities[0])[-top_k:][::-1]
    
    # Devolver los fragmentos más relevantes
    return [(text_fragments[i], similarities[0][i]) for i in most_similar_indices]

# Función para generar respuesta usando fragmentos relevantes
def generate_response_with_context(question, contexts):
    prompt = "\n\n".join([f"{context[0][:500]}... (Similitud: {context[1]:.4f})" for context in contexts])  # Limitar a 500 caracteres
    prompt = f"{prompt}\n\nPregunta: {question}\nRespuesta:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Ajustar max_new_tokens para una respuesta más corta
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Cargar o crear el vector store
text_fragments, fragment_embeddings = load_or_create_vector_store()

# Ejemplo de consulta
question = "¿Qué es el epitelio simple cuboidal?"
relevant_fragments = find_most_relevant_fragments(question, text_fragments, fragment_embeddings, top_k=3)

# Generar respuesta usando los fragmentos relevantes
response = generate_response_with_context(question, relevant_fragments)

print("Fragmentos relevantes:")
for fragment, similarity in relevant_fragments:
    print(f"Similitud: {similarity:.4f}, Fragmento: {fragment[:50]}...")  # Muestra los primeros 50 caracteres

print("Respuesta generada:", response)
