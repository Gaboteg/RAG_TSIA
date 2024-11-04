from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # usado para correr con el servidor de React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URL del endpoint del servicio que contiene tu modelo y RAG
RAG_SERVICE_URL = "http://localhost:8000/api/ask"  # Actualiza este puerto si es necesario

# Modelos de datos
class RunStatus(BaseModel):
    question: str
    response: str
    context: List[str]


class ThreadMessage(BaseModel):
    content: str
    role: str
    hidden: bool
    id: str
    created_at: int


class Thread(BaseModel):
    messages: List[ThreadMessage]


class CreateMessage(BaseModel):
    content: str


@app.post("/api/new")
async def post_new():
    """
    Inicia una nueva conversación preguntando al usuario qué está buscando.
    """
    # Esta función ahora solo devuelve un mensaje inicial en la interfaz
    return {"message": "¡Hola! Soy un asistente basado en RAG. ¿En qué puedo ayudarte hoy?"}


@app.post("/api/ask")
async def ask_question(message: CreateMessage):
    """
    Envía la pregunta del usuario al servicio de RAG y devuelve la respuesta.
    """
    question_payload = {"question": message.content}
    
    # Enviar la pregunta al servicio de RAG
    response = requests.post(RAG_SERVICE_URL, json=question_payload)
    
    if response.status_code == 200:
        result = response.json()
        return RunStatus(
            question=message.content,
            response=result["response"],
            context=result["context"]
        )
    else:
        return {"error": "No se pudo obtener una respuesta del servicio RAG"}


@app.get("/api/threads/{thread_id}")
async def get_thread(thread_id: str):
    """
    Obtiene la conversación actual. Para este ejemplo, solo devuelve un mensaje simulado.
    """
    # Simulación de mensajes anteriores para la UI
    messages = [
        ThreadMessage(
            content="Este es un mensaje simulado de la conversación.",
            role="assistant",
            hidden=False,
            id="1",
            created_at=0
        ),
        ThreadMessage(
            content="¿En qué más puedo ayudarte?",
            role="assistant",
            hidden=False,
            id="2",
            created_at=0
        ),
    ]

    return Thread(messages=messages)


@app.post("/api/threads/{thread_id}")
async def post_thread(thread_id: str, message: CreateMessage):
    """
    Envía un nuevo mensaje a la conversación existente y obtiene una respuesta del modelo.
    """
    # Reutilizar el endpoint `ask_question` para obtener la respuesta
    return await ask_question(message)
