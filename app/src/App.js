import React, { useState, useEffect } from 'react';
import './App.css';
import Header from "./components/Header";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import Loading from "./components/Loading";
import { useThread } from './hooks/useThread';
import { createThread, fetchThread, postMessage } from "./services/api";

function App() {
    const [threadId, setThreadId] = useState(null);
    const { messages, setActionMessages } = useThread(threadId); // Usa el hook para obtener mensajes
    const [loading, setLoading] = useState(true);

    // Función para crear un nuevo hilo
    const handleNewChat = async () => {
        setLoading(true);
        console.log("Creating new chat...");
        try {
            const response = await createThread();
            console.log("New thread created:", response);
            setThreadId(response.thread_id);
        } catch (error) {
            console.error("Error creating new chat:", error);
        } finally {
            setLoading(false);
        }
    };

    // Función para manejar el envío de mensajes
    const handleSendMessage = async (message) => {
        setLoading(true);
        console.log("Sending message:", message);
        try {
            await postMessage(threadId, { content: message }); // Envía el mensaje
            console.log("Message sent. Fetching messages...");
            await fetchMessages(); // Actualiza los mensajes después de enviar
        } catch (error) {
            console.error("Error sending message:", error);
        } finally {
            setLoading(false);
        }
    };

    // Función para obtener los mensajes del hilo actual
    const fetchMessages = async () => {
        if (threadId) {
            console.log(`Fetching messages for thread: ${threadId}`);
            try {
                const response = await fetchThread(threadId);
                console.log("Messages fetched:", response.messages); // Verifica lo que se recibe
                setActionMessages(response.messages); // Actualiza los mensajes
            } catch (error) {
                console.error("Error fetching messages:", error);
            }
        }
    };

    // UseEffect para manejar la creación de un nuevo hilo al inicio
    useEffect(() => {
        handleNewChat();
    }, []);
    
    // Renderizar la lista de mensajes
    console.log("Current messages:", messages);
    const messageList = messages
        .filter((message) => !message.hidden)
        .map((message) => (
            <ChatMessage
                message={message.content}
                role={message.role}
                key={message.id}
            />
        ));

    return (
        <div className="md:container md:mx-auto lg:px-32 h-screen bg-slate-700 flex flex-col">
            <Header onNewChat={handleNewChat} />
            <div className="flex flex-col grow overflow-scroll"> {/* Cambié flex-col-reverse a flex-col */}
                {loading && <Loading />}
                {messageList}
            </div>
            <div className="my-4">
                <ChatInput onSend={handleSendMessage} disabled={loading} />
            </div>
        </div>
    );
}



export default App;
