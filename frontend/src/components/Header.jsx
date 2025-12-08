import React from 'react';
import { Link } from 'react-router-dom';
import { useTheme } from '../hooks/useTheme';
import { Sun, Moon, Settings } from 'lucide-react';

function Header() {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="header">
      <Link to="/" className="logo">
        <img src="/makerspace-logo.png" alt="Makerspace" className="logo-img" />
        <div>
          <div className="logo-text">MAKERSPACE</div>
          <div className="logo-sub">Hogskolen i Ostfold</div>
        </div>
      </Link>
      <div className="header-actions">
        <button
          className="icon-btn"
          onClick={toggleTheme}
          title="Bytt tema"
          aria-label={`Bytt til ${theme === 'dark' ? 'lyst' : 'morkt'} tema`}
        >
          {theme === 'dark' ? <Sun size={24} /> : <Moon size={24} />}
        </button>
        <Link to="/admin" className="icon-btn" title="Admin" aria-label="Admin panel">
          <Settings size={24} />
        </Link>
      </div>
    </header>
  );
}

export default Header;
