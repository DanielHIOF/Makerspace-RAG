import React, { useState, useEffect, useRef } from 'react';
import { Copy, Check } from 'lucide-react';
import { formatMessage } from '../utils/formatMessage';

function Message({ message }) {
  const [copied, setCopied] = useState(false);
  const messageRef = useRef(null);

  // Initialize mermaid diagrams after render
  useEffect(() => {
    if (messageRef.current && window.mermaid) {
      const mermaidDivs = messageRef.current.querySelectorAll('.mermaid');
      mermaidDivs.forEach(div => {
        if (!div.dataset.processed) {
          window.mermaid.init(undefined, div);
          div.dataset.processed = 'true';
        }
      });
    }
  }, [message.content]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const roleClass = message.isUser ? 'user' : 'assistant';

  return (
    <div className={`message-wrapper ${roleClass}`}>
      <div
        className={`message ${roleClass}`}
        ref={messageRef}
        dangerouslySetInnerHTML={{ __html: formatMessage(message.content) }}
      />
      <div className="message-actions">
        <button
          className={`msg-action-btn ${copied ? 'copied' : ''}`}
          onClick={handleCopy}
          title="Kopier melding"
          aria-label="Kopier melding til utklippstavlen"
        >
          {copied ? <><Check size={14} /> Kopiert!</> : <><Copy size={14} /> Kopier</>}
        </button>
      </div>
    </div>
  );
}

export default Message;
