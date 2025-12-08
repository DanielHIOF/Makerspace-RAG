import React from 'react';
import { useChat } from '../hooks/useChat';
import { Printer, Zap, Cpu, Package } from 'lucide-react';

const categories = [
  { Icon: Printer, label: '3D-Printing', question: 'Jeg vil komme i gang med 3D-printing, hva trenger jeg a vite?' },
  { Icon: Zap, label: 'Laserkutter', question: 'Hvordan bruker jeg laserkutteren trygt?' },
  { Icon: Cpu, label: 'Elektronikk', question: 'Hjelp meg i gang med elektronikk og Arduino' },
  { Icon: Package, label: 'Materialer', question: 'Hvilke materialer kan jeg bruke i makerspacet?' }
];

function WelcomeMessage() {
  const { sendMessage } = useChat();

  return (
    <div className="welcome-message">
      <div className="welcome-icon">
        <img src="/makerspace-logo.png" alt="Makerspace" width="140" height="140" />
      </div>
      <h2>Hva skal vi lage i dag?</h2>
      <p>Din AI-assistent for 3D-printing, laserkutting, elektronikk og maker-prosjekter</p>

      <div className="category-grid" role="group" aria-label="Hurtigvalg kategorier">
        {categories.map((cat, index) => (
          <button
            key={index}
            className="category"
            onClick={() => sendMessage(cat.question)}
            aria-label={`Spor om ${cat.label}`}
          >
            <span className="cat-icon"><cat.Icon size={32} /></span>
            <span className="cat-label">{cat.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default WelcomeMessage;
