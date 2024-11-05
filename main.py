from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import pdfplumber
import faiss
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from rdflib import Graph, Namespace
from collections import defaultdict
import uuid
import re

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # usado para ejecutar con el servidor de React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de los modelos
model_name = "meta-llama/LLaMA-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="mitoken")
model = AutoModelForCausalLM.from_pretrained(model_name, token="mitoken")
embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Ruta para archivos
embeddings_path = './vector_store_faiss.index'
fragments_path = './vector_store_fragments.pkl'
pdf_path = './PDF-Books/LIBRO IFSSA Anatomia.y.Fisiologia.Humana.Marieb.pdf'
rdf_path = './human_body_data.rdf'  # Ruta del archivo RDF

# Función mejorada para extraer y segmentar texto de capítulos y subcapítulos
def extract_segmented_text_from_pdf(pdf_path):
    segmented_text = {}
    current_chapter = None
    current_subchapter = None
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split('\n'):
                    # Detectar títulos de capítulos (e.g., "1 INTRODUCCIÓN" o "CAPÍTULO 2 EL SISTEMA ESQUELÉTICO")
                    chapter_match = re.match(r'^\s*(CAPÍTULO\s+)?\d+\s+([A-Z\s]+)$', line)
                    subchapter_match = re.match(r'^\s*\d+\.\d+\s+([A-Z\s]+)', line)  # Detecta subcapítulos con "1.1", "2.1", etc.

                    # Si se detecta un capítulo, se crea una nueva entrada en el diccionario
                    if chapter_match:
                        current_chapter = chapter_match.group(2).strip()
                        segmented_text[current_chapter] = {}  # Inicializa como diccionario
                        current_subchapter = None
                    
                    # Si se detecta un subcapítulo, se añade como una clave dentro del capítulo actual
                    elif subchapter_match and current_chapter:
                        current_subchapter = subchapter_match.group(1).strip()
                        segmented_text[current_chapter][current_subchapter] = []  # Inicializa como lista
                    
                    # Si es una línea normal, añadirla al subcapítulo o capítulo actual
                    elif current_chapter:
                        # Segmentar el texto en oraciones para una mejor granularidad
                        sentences = re.split(r'(?<=\w[.!?])\s+', line)
                        
                        # Filtrar oraciones irrelevantes (e.g., de longitud corta o con palabras irrelevantes)
                        filtered_sentences = [s for s in sentences if len(s) > 50 and not re.match(r'^[\d\s]*$', s)]
                        
                        # Añadir a subcapítulo si está definido, sino al capítulo
                        if current_subchapter:
                            segmented_text[current_chapter][current_subchapter].extend(filtered_sentences)
                        else:
                            # Inicializar como lista si no hay subcapítulo actual
                            if current_chapter not in segmented_text:
                                segmented_text[current_chapter] = []
                            if isinstance(segmented_text[current_chapter], list):
                                segmented_text[current_chapter].extend(filtered_sentences)
    
    return segmented_text

# Cargar o crear el vector store
def load_or_create_vector_store() -> (List[str], faiss.IndexFlatL2):
    if os.path.exists(embeddings_path) and os.path.exists(fragments_path):
        index = faiss.read_index(embeddings_path)
        with open(fragments_path, 'rb') as f:
            paragraphs = pickle.load(f)
        print("Vector store cargado desde archivo.")
    else:
        # Extraer texto y segmentarlo
        segmented_text = extract_segmented_text_from_pdf(pdf_path)
        paragraphs = []
        for chapter, subchapters in segmented_text.items():
            for subchapter, sentences in subchapters.items():
                paragraphs.extend(sentences)

        embeddings = embedding_model.encode(paragraphs)

        # Crear FAISS index y almacenar embeddings
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(np.array(embeddings, dtype=np.float32))
        faiss.write_index(index, embeddings_path)

        # Guardar párrafos
        with open(fragments_path, 'wb') as f:
            pickle.dump(paragraphs, f)
        print("Vector store creado y guardado en archivo.")
    
    return paragraphs, index

# Cargar el RDF en un grafo
def load_knowledge_graph(rdf_path):
    graph = Graph()
    graph.parse(rdf_path, format='xml')
    return graph

def query_knowledge_graph(graph):
    # Definimos los namespaces que vamos a usar
    schema = Namespace("http://schema.org/")
    wd = Namespace("http://www.wikidata.org/entity/")
    
    # Consulta SPARQL adaptada a tu RDF
    sparql_query = """
    PREFIX schema1: <http://schema.org/>
    PREFIX ex: <http://example.org/>

    SELECT ?patientName ?patientAge ?conditionName ?systemAffectedName
    WHERE {
        ?patient ex:type "Pacient" ;
                 schema1:name ?patientName ;
                 schema1:age ?patientAge ;
                 ex:suffersFrom ?condition .

        ?condition ex:type "Condition" ;
                   schema1:name ?conditionName ;
                   ex:affects ?system .

        ?system schema1:name ?systemAffectedName .
    }
    """
    
    # Ejecutar la consulta
    results = graph.query(sparql_query)
    
    # Procesar los resultados en un formato de lista de diccionarios
    patients_data = []
    for row in results:
        patients_data.append({
            'patientName': str(row.patientName),
            'patientAge': str(row.patientAge),
            'conditionName': str(row.conditionName),
            'systemAffectedName': str(row.systemAffectedName),
        })
    
    return patients_data


# Obtener embedding de la pregunta
def get_question_embedding(question):
    return embedding_model.encode([question])[0]

# Encontrar fragmentos más relevantes usando FAISS
def find_most_relevant_fragments(question, sentences, index, top_k=3):
    question_embedding = get_question_embedding(question).astype(np.float32)
    _, top_indices = index.search(np.array([question_embedding]), top_k)
    return [sentences[i] for i in top_indices[0]]

def generate_response_with_context(question, context, kg_graph):
    # Detectar si la pregunta es sobre un paciente
    is_patient_query = any(keyword in question.lower() for keyword in ["nombre del paciente", "información del paciente", "datos del paciente", "tiene el paciente", "pasa al paciente", "que paciente", "sabes del paciente", "esta el paciente"])

    if is_patient_query:
        # Ejecutar la consulta en el Knowledge Graph
        results = query_knowledge_graph(kg_graph)
        
        # Formatear los resultados en un contexto para el modelo
        if results:
            patient_info = "Información de los pacientes:\n"
            for result in results:
                patient_info += (
                    f"Nombre: {result['patientName']} | "
                    f"Edad: {result['patientAge']} | "
                    f"Condición: {result['conditionName']} | "
                    f"Sistema afectado: {result['systemAffectedName']}\n"
                )
            
            # Crear un nuevo prompt usando el contexto de todos los pacientes
            prompt = (
                f"Contexto: {patient_info}\n\n"
                f"Pregunta: {question}\n\n"
                "Responde acerca del paciente basándote en el Contexto. Responde unicamente sobre lo que se te preguntó sin generar mas preguntas y respuestas:\nRespuesta:"
            )
            
            
            # Pasar el prompt al modelo para generar la respuesta
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Mover los tensores al dispositivo adecuado (CPU o GPU)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.4,
                    top_p=0.8,
                    top_k=50
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Calcular el índice donde comienza "Respuesta:"
            respuesta_index = len(prompt)  # Obtener la longitud de prompt

            # Retornar solo la parte de la respuesta que te interesa
            return response[respuesta_index:].strip()  # Retorna el texto después de "Respuesta:"


        else:
            return "No tengo información sobre el paciente solicitado."



    # Si no es una pregunta sobre un paciente, proceder con el contexto en el vector store
    prompt = (
        f"Contexto: {context}\n\n"
        f"Pregunta: {question}\n\n"
        f"Por favor solo responde a preguntas relacionadas con medicina, casos clinicos, biologia, si la pregunta no es referente a esos temas di 'Mil disculpas, no me especializo en esos temas'. "
        f"proporciona una respuesta detallada y precisa basada en el contexto."
        f"Si la respuesta a la Pregunta no se encuentra en el Contexto, responde 'No tengo la información para responder esa pregunta.' "
        f"Luego de dar la respuesta no generes otras preguntas o respuestas. Ni te repitas a ti mismo. "
        f" \nRespuesta:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Mover los tensores al dispositivo adecuado (CPU o GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.3,
            top_p=0.8,
            top_k=50
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Calcular el índice donde comienza "Respuesta:"
    respuesta_index = len(prompt)  # Obtener la longitud de prompt

    # Retornar solo la parte de la respuesta que te interesa
    return response[respuesta_index:].strip()  # Retorna el texto después de "Respuesta:"



        

# Estructura para almacenar los hilos en memoria
threads_store = defaultdict(list)

# Modelo para los mensajes
class CreateMessage(BaseModel):
    content: str

@app.post("/api/new")
async def post_new():
    thread_id = str(uuid.uuid4())  
    threads_store[thread_id] = []  
    return {"thread_id": thread_id}

@app.get("/api/threads/{thread_id}")
async def get_thread(thread_id: str):
    if thread_id not in threads_store:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return {"thread_id": thread_id, "messages": threads_store[thread_id]}

@app.post("/api/threads/{thread_id}")
async def post_thread(thread_id: str, message: CreateMessage):
    if thread_id not in threads_store:
        raise HTTPException(status_code=404, detail="Thread not found")

    new_message = {
        "content": message.content,
        "role": "user",
        "hidden": False,
        "id": str(len(threads_store[thread_id]))
    }
    threads_store[thread_id].append(new_message)

    # Cargar o crear vector store
    sentences, index = load_or_create_vector_store()

    # Cargar el Knowledge Graph
    kg_graph = load_knowledge_graph(rdf_path)

    question = message.content
    relevant_fragments = find_most_relevant_fragments(question, sentences, index, top_k=3)
    context = " ".join(relevant_fragments)
    response = generate_response_with_context(question, context, kg_graph)

    threads_store[thread_id].append({
        "content": response,
        "role": "assistant",
        "hidden": False,
        "id": str(len(threads_store[thread_id]))
    })

    return {"thread_id": thread_id, "messages": threads_store[thread_id]}