import { useState, useEffect } from 'react';
import { fetchThread } from '../services/api';

export const useThread = (threadId) => {
    const [messages, setMessages] = useState([]);

    const setActionMessages = (newMessages) => {
        setMessages(newMessages); // Actualiza los mensajes
    };

    const clearThread = () => {
        setMessages([]);
    };

    useEffect(() => {
        const fetchData = async () => {
            if (threadId) {
                try {
                    const response = await fetchThread(threadId);
                    setMessages(response.messages); // Actualiza los mensajes
                } catch (error) {
                    console.error("Error fetching messages:", error);
                }
            }
        };
        fetchData();
    }, [threadId]); // Se ejecuta cada vez que cambia threadId

    return { messages, setActionMessages, clearThread };
};
