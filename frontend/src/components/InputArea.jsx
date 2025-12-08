import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../hooks/useChat';
import { Send, Loader2 } from 'lucide-react';
import CommandSuggestions from './CommandSuggestions';

const commands = [
  { name: '/nybegynner', desc: 'Enkel forklaring for nybegynnere', value: '/nybegynner ' },
  { name: '/ekspert', desc: 'Teknisk og detaljert forklaring', value: '/ekspert ' },
  { name: '/norsk', desc: 'Svar pa norsk', value: '/norsk ' },
  { name: '/english', desc: 'Answer in English', value: '/english ' },
  { name: '/prusa', desc: 'Fokus pa Prusa-printere og PrusaSlicer', value: '/prusa ' },
  { name: '/3d', desc: 'Fokus pa 3D-printing generelt', value: '/3d ' },
  { name: '/laser', desc: 'Fokus pa laserkutting', value: '/laser ' },
  { name: '/cnc', desc: 'Fokus pa CNC-fresing', value: '/cnc ' },
  { name: '/elektronikk', desc: 'Fokus pa elektronikk og Arduino', value: '/elektronikk ' },
  { name: '/lodding', desc: 'Fokus pa lodding', value: '/lodding ' }
];

function InputArea() {
  const [input, setInput] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const inputRef = useRef(null);

  const {
    isLoading,
    selectedLevel,
    selectedLang,
    setSelectedLevel,
    setSelectedLang,
    sendMessage,
    getLevelDescription
  } = useChat();

  const levels = [
    { value: '/nybegynner ', label: '1', title: 'Enkel forklaring for nybegynnere' },
    { value: '', label: '2', title: 'Normal forklaring' },
    { value: '/ekspert ', label: '3', title: 'Teknisk og detaljert' }
  ];

  const languages = [
    { value: '', label: 'Auto', title: 'Automatisk sprakgjenkjenning' },
    { value: '/norsk ', label: 'NO', title: 'Svar pa norsk' },
    { value: '/english ', label: 'EN', title: 'Answer in English' }
  ];

  // Filter commands based on input
  const filteredCommands = React.useMemo(() => {
    const lastSlashIndex = input.lastIndexOf('/');
    if (lastSlashIndex === -1) return [];
    const query = input.substring(lastSlashIndex + 1).toLowerCase();
    return commands.filter(cmd => cmd.name.toLowerCase().includes(query));
  }, [input]);

  const handleInputChange = (e) => {
    const value = e.target.value;
    setInput(value);

    if (value.includes('/')) {
      setShowSuggestions(true);
      setSelectedIndex(-1);
    } else {
      setShowSuggestions(false);
    }
  };

  const handleKeyDown = (e) => {
    // Ctrl+Enter or Cmd+Enter always sends
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSend();
      return;
    }

    if (!showSuggestions || filteredCommands.length === 0) {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSend();
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => Math.min(prev + 1, filteredCommands.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => Math.max(prev - 1, -1));
    } else if (e.key === 'Enter' && selectedIndex >= 0) {
      e.preventDefault();
      insertCommand(filteredCommands[selectedIndex].value);
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      handleSend();
    }
  };

  const insertCommand = (commandValue) => {
    const lastSlashIndex = input.lastIndexOf('/');
    if (lastSlashIndex !== -1) {
      setInput(input.substring(0, lastSlashIndex) + commandValue);
    } else {
      setInput(commandValue);
    }
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      sendMessage(input.trim());
      setInput('');
    }
  };

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <footer className="input-area">
      <div className="level-desc-bar" aria-live="polite">
        {getLevelDescription()}
      </div>

      <div className="input-container">
        <div className="side-panel left">
          <div className="control-group">
            <label id="level-label">Niva</label>
            <div className="btn-group" role="group" aria-labelledby="level-label">
              {levels.map((level) => (
                <button
                  key={level.value}
                  className={`ctrl-btn ${selectedLevel === level.value ? 'active' : ''}`}
                  onClick={() => setSelectedLevel(level.value)}
                  title={level.title}
                  aria-pressed={selectedLevel === level.value}
                >
                  {level.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="input-row">
          <div className="input-wrapper">
            <input
              ref={inputRef}
              type="text"
              id="input"
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Skriv ditt sporsmal her... (Ctrl+Enter for a sende)"
              autoComplete="off"
              aria-label="Skriv melding"
            />
            {showSuggestions && filteredCommands.length > 0 && (
              <CommandSuggestions
                commands={filteredCommands}
                selectedIndex={selectedIndex}
                onSelect={insertCommand}
              />
            )}
          </div>
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            aria-label="Send melding"
          >
            {isLoading ? <Loader2 size={20} className="loading" /> : <Send size={20} />}
          </button>
        </div>

        <div className="side-panel right">
          <div className="control-group">
            <label id="lang-label">Sprak</label>
            <div className="btn-group" role="group" aria-labelledby="lang-label">
              {languages.map((lang) => (
                <button
                  key={lang.value}
                  className={`lang-btn ${selectedLang === lang.value ? 'active' : ''}`}
                  onClick={() => setSelectedLang(lang.value)}
                  title={lang.title}
                  aria-pressed={selectedLang === lang.value}
                >
                  {lang.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="brand-bar" aria-hidden="true"></div>
    </footer>
  );
}

export default InputArea;
