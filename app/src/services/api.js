import axios from 'axios';

const API_URL = 'http://localhost:8000/api'; // Cambia esto a tu URL de la API

export const createThread = async () => {
    const response = await axios.post(`${API_URL}/new`);
    return response.data;
};

export const fetchThread = async (threadId) => {
    const response = await axios.get(`${API_URL}/threads/${threadId}`);
    return response.data;
};

export const postMessage = async (threadId, message) => {
    try {
        console.log("Posting message to API:", { content: message.content }); // Ver lo que se envía
        const response = await axios.post(`http://localhost:8000/api/threads/${threadId}`, {
            content: message.content,
        });
        return response.data;
    } catch (error) {
        console.error("Error posting message:", error.response ? error.response.data : error.message);
        throw error; // Re-lanzar el error para manejarlo más adelante
    }
};