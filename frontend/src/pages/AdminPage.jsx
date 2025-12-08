import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useTheme } from '../hooks/useTheme';
import * as adminApi from '../services/adminApi';
import ComponentsManager from '../components/ComponentsManager';
import {
  MessageCircle, Sun, Moon, Upload, Package, FileText, RefreshCw,
  Check, X, AlertTriangle, Sparkles, Book, Shield, Wrench, Settings,
  FolderOpen, Link as LinkIcon, Home, ChevronLeft, Loader2, Database,
  HardDrive, FileSpreadsheet, Plus, Trash2, Edit3, Lock, LogOut
} from 'lucide-react';
import '../styles/admin.css';

function AdminPage() {
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('uploads');
  const [stats, setStats] = useState({ chunks: 0, size_kb: 0 });
  const [reloadStatus, setReloadStatus] = useState('Klikk etter å ha lagt til innhold');
  const [isReloading, setIsReloading] = useState(false);

  // Auth state
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginPassword, setLoginPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  // File upload state
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadMessage, setUploadMessage] = useState({ text: '', type: '' });
  const fileInputRef = useRef(null);

  // PDF preview state
  const [pdfPreview, setPdfPreview] = useState({ active: false, content: '', info: '' });
  const [docContext, setDocContext] = useState('');
  const [docCategory, setDocCategory] = useState('vault');
  const [enhancedPreview, setEnhancedPreview] = useState({ active: false, content: '', info: '' });
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [enhanceSeconds, setEnhanceSeconds] = useState(0);

  // XLSX preview state
  const [xlsxPreview, setXlsxPreview] = useState({ active: false, items: [], duplicates: [], info: '' });
  const [xlsxChecked, setXlsxChecked] = useState([]);

  // Text input state
  const [textInput, setTextInput] = useState('');
  const [textMessage, setTextMessage] = useState({ text: '', type: '' });

  // Check auth on mount
  useEffect(() => {
    checkAuth();
  }, []);

  // Load stats when logged in
  useEffect(() => {
    if (isLoggedIn) {
      loadStats();
    }
  }, [isLoggedIn]);

  const checkAuth = async () => {
    try {
      const response = await fetch('/auth/check', {
        credentials: 'include'
      });
      if (response.ok) {
        const data = await response.json();
        setIsLoggedIn(data.authenticated);
      }
    } catch (error) {
      console.error('Auth check failed:', error);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setIsLoggingIn(true);
    setLoginError('');

    try {
      const response = await fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: loginPassword }),
        credentials: 'include'
      });

      const data = await response.json();

      if (data.success) {
        setIsLoggedIn(true);
        setLoginPassword('');
      } else {
        setLoginError(data.error || 'Feil passord');
      }
    } catch (error) {
      console.error('Login error:', error);
      setLoginError('Nettverksfeil - sjekk at serveren kjører');
    }

    setIsLoggingIn(false);
  };

  const handleLogout = async () => {
    try {
      await fetch('/logout', {
        method: 'POST',
        credentials: 'include'
      });
      navigate('/');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const loadStats = async () => {
    try {
      const data = await adminApi.getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  // File handling
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    if (droppedFiles.length) handleFiles(droppedFiles);
  };

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length) handleFiles(selectedFiles);
  };

  const handleFiles = async (newFiles) => {
    setPdfPreview({ active: false, content: '', info: '' });
    setEnhancedPreview({ active: false, content: '', info: '' });
    setXlsxPreview({ active: false, items: [], duplicates: [], info: '' });
    setUploadMessage({ text: '', type: '' });

    const fileStates = newFiles.map((f, i) => ({
      id: i,
      name: f.name,
      status: 'pending',
      statusText: 'Venter...'
    }));
    setFiles(fileStates);

    for (let i = 0; i < newFiles.length; i++) {
      const file = newFiles[i];
      setFiles(prev => prev.map(f =>
        f.id === i ? { ...f, status: 'processing', statusText: 'Behandler...' } : f
      ));

      const ext = file.name.toLowerCase().split('.').pop();

      try {
        if (ext === 'pdf') {
          await handlePDFFile(file, i);
        } else if (ext === 'xlsx') {
          await handleXLSXFile(file, i);
        } else {
          await handleGenericUpload(file, i);
        }
      } catch (error) {
        setFiles(prev => prev.map(f =>
          f.id === i ? { ...f, status: 'error', statusText: `Feil: ${error.message}` } : f
        ));
      }
    }
  };

  const handlePDFFile = async (file, index) => {
    const data = await adminApi.extractPDF(file);
    if (data.success) {
      setPdfPreview({
        active: true,
        content: data.text,
        info: `${data.filename} | ${data.page_count} sider | ${data.char_count} tegn`
      });
      setFiles(prev => prev.map(f =>
        f.id === index ? { ...f, status: 'success', statusText: 'Ekstrahert' } : f
      ));
    } else {
      throw new Error(data.error);
    }
  };

  const handleXLSXFile = async (file, index) => {
    const data = await adminApi.extractXLSX(file);
    if (data.success) {
      const items = data.items_to_add || [];
      setXlsxPreview({
        active: true,
        items: items,
        duplicates: data.duplicates_skipped || [],
        info: `${data.filename} - ${data.total_rows} rader, ${items.length} nye`
      });
      setXlsxChecked(items.map((_, i) => i));
      setFiles(prev => prev.map(f =>
        f.id === index ? { ...f, status: 'success', statusText: 'Parset' } : f
      ));
    } else {
      throw new Error(data.error);
    }
  };

  const handleGenericUpload = async (file, index) => {
    const data = await adminApi.uploadFile(file);
    if (data.success) {
      setFiles(prev => prev.map(f =>
        f.id === index ? { ...f, status: 'success', statusText: 'Lastet opp' } : f
      ));
      if (data.stats) setStats(data.stats);
    } else {
      throw new Error(data.error);
    }
  };

  // PDF actions
  const handleEnhance = async () => {
    if (!pdfPreview.content.trim()) {
      setUploadMessage({ text: 'Ingen tekst å behandle', type: 'error' });
      return;
    }
    if (!docContext.trim()) {
      setUploadMessage({ text: 'Beskriv hva dokumentet handler om', type: 'error' });
      return;
    }

    setIsEnhancing(true);
    setEnhanceSeconds(0);
    const timer = setInterval(() => setEnhanceSeconds(s => s + 1), 1000);

    try {
      const data = await adminApi.enhancePDF(pdfPreview.content, docContext, docCategory);
      clearInterval(timer);

      if (data.success) {
        setEnhancedPreview({
          active: true,
          content: data.enhanced_text,
          info: `${data.original_chars} → ${data.enhanced_chars} tegn`
        });
        setPdfPreview(prev => ({ ...prev, active: false }));
      } else {
        setUploadMessage({ text: data.error, type: 'error' });
      }
    } catch (error) {
      clearInterval(timer);
      setUploadMessage({ text: error.message, type: 'error' });
    }

    setIsEnhancing(false);
  };

  const handleAcceptContent = async (content) => {
    try {
      const data = await adminApi.approveSummary(content, docCategory);
      if (data.success) {
        setUploadMessage({ text: data.message + ' - Husk å laste inn søkeindeks!', type: 'success' });
        setPdfPreview({ active: false, content: '', info: '' });
        setEnhancedPreview({ active: false, content: '', info: '' });
        if (data.stats) setStats(data.stats);
      } else {
        setUploadMessage({ text: data.error, type: 'error' });
      }
    } catch (error) {
      setUploadMessage({ text: error.message, type: 'error' });
    }
  };

  // XLSX actions
  const handleXlsxCheckChange = (idx) => {
    setXlsxChecked(prev =>
      prev.includes(idx) ? prev.filter(i => i !== idx) : [...prev, idx]
    );
  };

  const handleApproveXlsx = async () => {
    const itemsToSend = xlsxChecked.map(idx => xlsxPreview.items[idx]);
    if (itemsToSend.length === 0) {
      setUploadMessage({ text: 'Ingen komponenter valgt', type: 'error' });
      return;
    }

    try {
      const data = await adminApi.approveXLSX(itemsToSend);
      if (data.success) {
        setUploadMessage({ text: `${data.message} - Husk å laste inn søkeindeks!`, type: 'success' });
        setXlsxPreview({ active: false, items: [], duplicates: [], info: '' });
      } else {
        setUploadMessage({ text: data.error, type: 'error' });
      }
    } catch (error) {
      setUploadMessage({ text: error.message, type: 'error' });
    }
  };

  // Text input
  const handleAddText = async () => {
    if (!textInput.trim()) {
      setTextMessage({ text: 'Vennligst skriv inn tekst', type: 'error' });
      return;
    }

    try {
      const data = await adminApi.addText(textInput);
      if (data.success) {
        setTextMessage({ text: data.message, type: 'success' });
        setTextInput('');
        if (data.stats) setStats(data.stats);
      } else {
        setTextMessage({ text: data.error, type: 'error' });
      }
    } catch (error) {
      setTextMessage({ text: error.message, type: 'error' });
    }
  };

  // Reload embeddings
  const handleReload = async () => {
    setIsReloading(true);
    setReloadStatus('Laster inn...');

    try {
      const data = await adminApi.reloadIndex();
      if (data.success) {
        setReloadStatus(`Lastet ${data.chunks} biter`);
      } else {
        setReloadStatus('Innlasting feilet');
      }
    } catch (error) {
      setReloadStatus('Feil: ' + error.message);
    }

    setIsReloading(false);
  };

  // Login screen
  if (!isLoggedIn) {
    return (
      <div className="admin-page">
        <header className="header">
          <Link to="/" className="logo">
            <img src="/makerspace-logo.png" alt="Makerspace" className="logo-img" />
            <div>
              <div className="logo-text">MAKERSPACE</div>
              <div className="logo-sub">Admin Panel</div>
            </div>
          </Link>
          <div className="header-actions">
            <button className="icon-btn" onClick={toggleTheme} title="Bytt tema">
              {theme === 'dark' ? <Sun size={24} /> : <Moon size={24} />}
            </button>
          </div>
        </header>

        <div className="login-container">
          <div className="login-card">
            <div className="login-icon">
              <Lock size={48} />
            </div>
            <h2>Admin Login</h2>
            <p>Skriv inn admin-passordet for å fortsette</p>

            <form onSubmit={handleLogin}>
              <input
                type="password"
                value={loginPassword}
                onChange={(e) => setLoginPassword(e.target.value)}
                placeholder="Passord"
                className="input-field"
                autoFocus
              />
              {loginError && (
                <div className="message error">
                  <AlertTriangle size={18} /> {loginError}
                </div>
              )}
              <button type="submit" className="btn btn-primary" disabled={isLoggingIn}>
                {isLoggingIn ? <Loader2 size={18} className="loading" /> : <Lock size={18} />}
                Logg inn
              </button>
            </form>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="admin-page">
      <header className="header">
        <Link to="/" className="logo">
          <img src="/makerspace-logo.png" alt="Makerspace" className="logo-img" />
          <div>
            <div className="logo-text">MAKERSPACE</div>
            <div className="logo-sub">Admin Panel</div>
          </div>
        </Link>
        <div className="header-actions">
          <button className="icon-btn" onClick={toggleTheme} title="Bytt tema">
            {theme === 'dark' ? <Sun size={24} /> : <Moon size={24} />}
          </button>
          <button className="icon-btn" onClick={handleLogout} title="Logg ut">
            <LogOut size={24} />
          </button>
        </div>
      </header>

      <div className="admin-container">
        <div className="page-title">
          <h1>Administrer kunnskapsbasen</h1>
          <p className="subtitle">Last opp filer og administrer komponenter</p>
        </div>

        <div className="stats-bar">
          <div className="stat-card">
            <div className="stat-value">{stats.chunks}</div>
            <div className="stat-label"><Database size={14} /> Kunnskapsbiter</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.size_kb} KB</div>
            <div className="stat-label"><HardDrive size={14} /> Database størrelse</div>
          </div>
          <div className="stat-card">
            <button className="btn" onClick={handleReload} disabled={isReloading}>
              {isReloading ? <Loader2 size={18} className="loading" /> : <RefreshCw size={18} />}
              Last inn embeddings
            </button>
            <div className="stat-label">{reloadStatus}</div>
          </div>
        </div>

        {/* Tabs */}
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'uploads' ? 'active' : ''}`}
            onClick={() => setActiveTab('uploads')}
          >
            <Upload size={18} /> Uploads
          </button>
          <button
            className={`tab ${activeTab === 'components' ? 'active' : ''}`}
            onClick={() => setActiveTab('components')}
          >
            <Package size={18} /> Komponenter
          </button>
        </div>

        {/* Uploads Tab */}
        {activeTab === 'uploads' && (
          <div className="tab-content active">
            {/* Upload Zone */}
            <div
              className={`upload-zone ${isDragging ? 'dragover' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="icon"><Upload size={48} /></div>
              <p>Dra og slipp filer her eller klikk for å velge</p>
              <span className="formats">Støtter: PDF, XLSX (auto-detektert)</span>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept=".pdf,.xlsx"
                multiple
                style={{ display: 'none' }}
              />
            </div>

            {/* File List */}
            {files.length > 0 && (
              <div className="file-list">
                {files.map(file => (
                  <div key={file.id} className={`file-item ${file.status}`}>
                    <span className="file-name">
                      <FileText size={18} /> {file.name}
                    </span>
                    <span className="file-status">
                      {file.status === 'success' && <Check size={16} />}
                      {file.status === 'error' && <X size={16} />}
                      {file.status === 'processing' && <Loader2 size={16} className="loading" />}
                      {file.statusText}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Upload Message */}
            {uploadMessage.text && (
              <div className={`message ${uploadMessage.type}`}>
                {uploadMessage.type === 'success' ? <Check size={18} /> : <AlertTriangle size={18} />}
                {uploadMessage.text}
              </div>
            )}

            {/* PDF Preview */}
            {pdfPreview.active && (
              <div className="preview-section active">
                <div className="preview-header">
                  <h3><FileText size={20} /> PDF Ekstrahering</h3>
                  <span className="preview-info">{pdfPreview.info}</span>
                </div>
                <textarea
                  value={pdfPreview.content}
                  onChange={(e) => setPdfPreview(prev => ({ ...prev, content: e.target.value }))}
                  style={{ height: '300px', fontFamily: 'monospace', fontSize: '0.85rem' }}
                />

                <div className="context-box">
                  <label><Sparkles size={16} /> Hva handler dokumentet om?</label>
                  <input
                    type="text"
                    className="input-field"
                    value={docContext}
                    onChange={(e) => setDocContext(e.target.value)}
                    placeholder="F.eks: 'Epilog laser bruksanvisning', 'HMS-regler for 3D-printing'"
                  />
                  <div className="tag-row">
                    <button className="quick-tag" onClick={() => setDocContext('Utstyrsmanual')}>
                      <Book size={14} /> Manual
                    </button>
                    <button className="quick-tag" onClick={() => setDocContext('HMS og sikkerhet')}>
                      <Shield size={14} /> HMS
                    </button>
                    <button className="quick-tag" onClick={() => setDocContext('Feilsøkingsguide')}>
                      <Wrench size={14} /> Feilsøking
                    </button>
                    <button className="quick-tag" onClick={() => setDocContext('Materialinnstillinger')}>
                      <Settings size={14} /> Innstillinger
                    </button>
                  </div>

                  <div className="section-divider">
                    <label><FolderOpen size={16} /> Hvor skal innholdet lagres?</label>
                    <select
                      className="select-field"
                      value={docCategory}
                      onChange={(e) => setDocCategory(e.target.value)}
                    >
                      <option value="vault">Generell kunnskap (vault.txt)</option>
                      <option value="utstyr">Utstyr (utstyr.json)</option>
                      <option value="regler">Regler (regler.json)</option>
                      <option value="rom">Rom (rom.json)</option>
                      <option value="ressurser">Ressurser (ressurser.json)</option>
                    </select>
                  </div>
                </div>

                <div className="preview-actions">
                  <button className="btn" onClick={handleEnhance} disabled={isEnhancing}>
                    {isEnhancing ? (
                      <><Loader2 size={18} className="loading" /> AI jobber... {enhanceSeconds}s</>
                    ) : (
                      <><Sparkles size={18} /> Strukturer med AI</>
                    )}
                  </button>
                  <button className="btn btn-success" onClick={() => handleAcceptContent(pdfPreview.content)}>
                    <Check size={18} /> Godkjenn rå tekst
                  </button>
                  <button className="btn btn-secondary" onClick={() => setPdfPreview({ active: false, content: '', info: '' })}>
                    <X size={18} /> Avbryt
                  </button>
                </div>
              </div>
            )}

            {/* Enhanced Preview */}
            {enhancedPreview.active && (
              <div className="preview-section active">
                <div className="preview-header">
                  <h3><Sparkles size={20} /> AI-strukturert tekst</h3>
                  <span className="preview-info">{enhancedPreview.info}</span>
                </div>
                <textarea
                  value={enhancedPreview.content}
                  onChange={(e) => setEnhancedPreview(prev => ({ ...prev, content: e.target.value }))}
                  style={{ height: '350px', fontFamily: 'monospace', fontSize: '0.85rem' }}
                />
                <div className="preview-actions">
                  <button className="btn btn-success" onClick={() => handleAcceptContent(enhancedPreview.content)}>
                    <Check size={18} /> Godkjenn
                  </button>
                  <button className="btn btn-secondary" onClick={() => {
                    setEnhancedPreview({ active: false, content: '', info: '' });
                    setPdfPreview(prev => ({ ...prev, active: true }));
                  }}>
                    <ChevronLeft size={18} /> Tilbake til rå tekst
                  </button>
                  <button className="btn btn-secondary" onClick={() => setEnhancedPreview({ active: false, content: '', info: '' })}>
                    <X size={18} /> Avbryt
                  </button>
                </div>
              </div>
            )}

            {/* XLSX Preview */}
            {xlsxPreview.active && (
              <div className="preview-section active">
                <div className="preview-header">
                  <h3><FileSpreadsheet size={20} /> Excel Import</h3>
                  <span className="preview-info">{xlsxPreview.info}</span>
                </div>

                {xlsxPreview.duplicates.length > 0 && (
                  <div className="info-box warning">
                    <AlertTriangle size={18} /> {xlsxPreview.duplicates.length} duplikater filtrert bort
                  </div>
                )}

                <div className="table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th><Check size={16} /></th>
                        <th>Navn</th>
                        <th>Lokasjon</th>
                        <th>Kategori</th>
                      </tr>
                    </thead>
                    <tbody>
                      {xlsxPreview.items.map((item, idx) => (
                        <tr key={idx}>
                          <td>
                            <input
                              type="checkbox"
                              checked={xlsxChecked.includes(idx)}
                              onChange={() => handleXlsxCheckChange(idx)}
                            />
                          </td>
                          <td>{item.name}</td>
                          <td>{item.location || '-'}</td>
                          <td>{item.category || 'other'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="preview-actions">
                  <button className="btn btn-success" onClick={handleApproveXlsx}>
                    <Plus size={18} /> Legg til {xlsxChecked.length} komponenter
                  </button>
                  <button className="btn btn-secondary" onClick={() => setXlsxPreview({ active: false, items: [], duplicates: [], info: '' })}>
                    <X size={18} /> Avbryt
                  </button>
                </div>
              </div>
            )}

            {/* Text Input */}
            <div style={{ marginTop: '30px' }}>
              <h3 style={{ color: 'var(--brand)', marginBottom: '15px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Edit3 size={20} /> Legg til tekst direkte
              </h3>
              <textarea
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Lim inn eller skriv tekst her. Den blir automatisk delt opp i biter for kunnskapsbasen..."
              />
              <button className="btn" onClick={handleAddText}>
                <Plus size={18} /> Legg til i kunnskapsbasen
              </button>
              {textMessage.text && (
                <div className={`message ${textMessage.type}`}>
                  {textMessage.type === 'success' ? <Check size={18} /> : <AlertTriangle size={18} />}
                  {textMessage.text}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Components Tab */}
        {activeTab === 'components' && (
          <div className="tab-content active">
            <ComponentsManager />
          </div>
        )}
      </div>
    </div>
  );
}

export default AdminPage;
