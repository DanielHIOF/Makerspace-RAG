import React, { createContext, useContext, useState, useCallback } from 'react';
import { sendChatMessage } from '../services/api';

const ChatContext = createContext();

export function ChatProvider({ children }) {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [summary, setSummary] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedLevel, setSelectedLevel] = useState('');
  const [selectedLang, setSelectedLang] = useState('');

  const levelDescriptions = {
    '/nybegynner ': 'Nybegynner: Enkelt sprak, steg-for-steg, ingen faguttrykk',
    '': 'Normal: Balansert forklaring med korte definisjoner av faguttrykk',
    '/ekspert ': 'Ekspert: Teknisk presisjon, fagterminologi, avanserte detaljer'
  };

  const addMessage = useCallback((content, isUser) => {
    const newMessage = {
      id: Date.now(),
      content,
      isUser,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, newMessage]);

    // Update history for context
    setHistory(prev => {
      const updated = [...prev, { role: isUser ? 'user' : 'assistant', content }];
      // Keep last 60 messages
      return updated.slice(-60);
    });

    return newMessage;
  }, []);

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || isLoading) return;

    const fullMessage = selectedLang + selectedLevel + text;
    addMessage(text, true);
    setIsLoading(true);

    try {
      const data = await sendChatMessage(
        fullMessage,
        history.slice(0, -1), // Exclude the message we just added
        summary
      );

      if (data.summary) {
        setSummary(data.summary);
      }

      addMessage(data.response, false);
    } catch (error) {
      console.error('Chat error:', error);
      addMessage('Beklager, noe gikk galt. Prov igjen.', false);
    } finally {
      setIsLoading(false);
    }
  }, [selectedLang, selectedLevel, history, summary, isLoading, addMessage]);

  const newChat = useCallback(() => {
    setMessages([]);
    setHistory([]);
    setSummary('');
  }, []);

  const getLevelDescription = useCallback(() => {
    return levelDescriptions[selectedLevel] || levelDescriptions[''];
  }, [selectedLevel]);

  return (
    <ChatContext.Provider value={{
      messages,
      isLoading,
      selectedLevel,
      selectedLang,
      setSelectedLevel,
      setSelectedLang,
      sendMessage,
      newChat,
      getLevelDescription
    }}>
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}
