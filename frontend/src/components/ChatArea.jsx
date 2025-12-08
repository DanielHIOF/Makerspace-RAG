import React, { useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChat';
import Message from './Message';
import WelcomeMessage from './WelcomeMessage';
import LoadingIndicator from './LoadingIndicator';

function ChatArea() {
  const { messages, isLoading } = useChat();
  const chatAreaRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const hasMessages = messages.length > 0;

  return (
    <main className="chat-area" ref={chatAreaRef}>
      {!hasMessages && <WelcomeMessage />}

      <div className="messages" role="log" aria-live="polite" aria-label="Chat meldinger">
        {messages.map((msg) => (
          <Message key={msg.id} message={msg} />
        ))}
        {isLoading && <LoadingIndicator />}
      </div>
    </main>
  );
}

export default ChatArea;
