import React, { useEffect, useRef } from 'react';

function CommandSuggestions({ commands, selectedIndex, onSelect }) {
  const listRef = useRef(null);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current && selectedIndex >= 0) {
      const items = listRef.current.querySelectorAll('.command-item');
      items[selectedIndex]?.scrollIntoView({ block: 'nearest' });
    }
  }, [selectedIndex]);

  return (
    <div
      className="command-suggestions"
      ref={listRef}
      role="listbox"
      aria-label="Kommando forslag"
    >
      {commands.map((cmd, index) => (
        <div
          key={cmd.name}
          className={`command-item ${index === selectedIndex ? 'selected' : ''}`}
          onClick={() => onSelect(cmd.value)}
          role="option"
          aria-selected={index === selectedIndex}
        >
          <div className="command-name">{cmd.name}</div>
          <div className="command-desc">{cmd.desc}</div>
        </div>
      ))}
    </div>
  );
}

export default CommandSuggestions;
