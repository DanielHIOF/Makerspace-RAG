import React, { useState, useEffect, useCallback } from 'react';
import {
  Search, Plus, Edit3, Trash2, Check, X, AlertTriangle,
  Package, MapPin, RefreshCw, Loader2, ShoppingCart, Tag
} from 'lucide-react';

function ComponentsManager() {
  const [components, setComponents] = useState([]);
  const [filteredComponents, setFilteredComponents] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filter state
  const [showRestockOnly, setShowRestockOnly] = useState(false);
  const [hylleplasser, setHylleplasser] = useState([]);
  const [selectedHylleplass, setSelectedHylleplass] = useState('');
  const [kategorier, setKategorier] = useState([]);
  const [selectedKategori, setSelectedKategori] = useState('');

  // Edit state
  const [editingId, setEditingId] = useState(null);
  const [editForm, setEditForm] = useState({});

  // Add new component state
  const [showAddForm, setShowAddForm] = useState(false);
  const [newComponent, setNewComponent] = useState({
    name: '',
    hylleplass: '',
    kategori: 'Annet',
    forbruksvare: false,
    restock: false,
    antall: 0
  });

  // Toast notification
  const [toast, setToast] = useState({ show: false, message: '', type: '' });

  const showToast = (message, type = 'success') => {
    setToast({ show: true, message, type });
    setTimeout(() => setToast({ show: false, message: '', type: '' }), 3000);
  };

  // Load components
  const loadComponents = useCallback(async () => {
    setLoading(true);
    try {
      let url = '/api/components';
      const params = new URLSearchParams();

      if (searchQuery) params.append('q', searchQuery);
      if (showRestockOnly) params.append('restock', 'true');

      if (params.toString()) url += '?' + params.toString();

      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to fetch components');

      const data = await response.json();
      setComponents(data);
      setFilteredComponents(data);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error loading components:', err);
    } finally {
      setLoading(false);
    }
  }, [searchQuery, showRestockOnly]);

  // Load hylleplasser
  const loadHylleplasser = async () => {
    try {
      const response = await fetch('/api/hylleplasser');
      if (response.ok) {
        const data = await response.json();
        setHylleplasser(data);
      }
    } catch (err) {
      console.error('Error loading hylleplasser:', err);
    }
  };

  // Load kategorier
  const loadKategorier = async () => {
    try {
      const response = await fetch('/api/kategorier');
      if (response.ok) {
        const data = await response.json();
        setKategorier(data);
      }
    } catch (err) {
      console.error('Error loading kategorier:', err);
    }
  };

  useEffect(() => {
    loadComponents();
    loadHylleplasser();
    loadKategorier();
  }, [loadComponents]);

  // Filter by hylleplass and kategori
  useEffect(() => {
    let filtered = components;
    if (selectedHylleplass) {
      filtered = filtered.filter(c => c.hylleplass === selectedHylleplass);
    }
    if (selectedKategori) {
      filtered = filtered.filter(c => c.kategori === selectedKategori);
    }
    setFilteredComponents(filtered);
  }, [selectedHylleplass, selectedKategori, components]);

  // Handle search
  const handleSearch = (e) => {
    e.preventDefault();
    loadComponents();
  };

  // Add component
  const handleAddComponent = async () => {
    if (!newComponent.name.trim() || !newComponent.hylleplass.trim()) {
      showToast('Navn og hylleplass er påkrevd', 'error');
      return;
    }

    try {
      const response = await fetch('/api/components', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newComponent)
      });

      if (response.ok) {
        showToast('Komponent lagt til!');
        setNewComponent({ name: '', hylleplass: '', kategori: 'Annet', forbruksvare: false, restock: false, antall: 0 });
        setShowAddForm(false);
        loadComponents();
        loadHylleplasser();
        loadKategorier();
      } else {
        const data = await response.json();
        showToast(data.error || 'Feil ved lagring', 'error');
      }
    } catch (err) {
      showToast('Nettverksfeil', 'error');
    }
  };

  // Edit component
  const startEdit = (component) => {
    setEditingId(component.id);
    setEditForm({ ...component });
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditForm({});
  };

  const saveEdit = async () => {
    try {
      const response = await fetch(`/api/components/${editingId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editForm)
      });

      if (response.ok) {
        showToast('Komponent oppdatert!');
        setEditingId(null);
        loadComponents();
      } else {
        const data = await response.json();
        showToast(data.error || 'Feil ved oppdatering', 'error');
      }
    } catch (err) {
      showToast('Nettverksfeil', 'error');
    }
  };

  // Delete component
  const handleDelete = async (id, name) => {
    if (!window.confirm(`Slette "${name}"?`)) return;

    try {
      const response = await fetch(`/api/components/${id}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        showToast('Komponent slettet!');
        loadComponents();
      } else {
        showToast('Feil ved sletting', 'error');
      }
    } catch (err) {
      showToast('Nettverksfeil', 'error');
    }
  };

  // Toggle restock
  const toggleRestock = async (component) => {
    try {
      const response = await fetch(`/api/components/${component.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...component, restock: !component.restock })
      });

      if (response.ok) {
        loadComponents();
      }
    } catch (err) {
      console.error('Error toggling restock:', err);
    }
  };

  return (
    <div className="components-manager">
      {/* Toast */}
      {toast.show && (
        <div className={`toast ${toast.type}`}>
          {toast.type === 'success' ? <Check size={18} /> : <AlertTriangle size={18} />}
          {toast.message}
        </div>
      )}

      {/* Toolbar */}
      <div className="toolbar">
        <form onSubmit={handleSearch} className="search-form">
          <div className="search-input-wrapper">
            <Search size={18} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Søk etter komponenter..."
              className="input-field"
            />
          </div>
          <button type="submit" className="btn">
            <Search size={18} /> Søk
          </button>
        </form>

        <div className="toolbar-actions">
          <select
            value={selectedKategori}
            onChange={(e) => setSelectedKategori(e.target.value)}
            className="select-field"
          >
            <option value="">Alle kategorier</option>
            {kategorier.map(k => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>

          <select
            value={selectedHylleplass}
            onChange={(e) => setSelectedHylleplass(e.target.value)}
            className="select-field"
          >
            <option value="">Alle hyller</option>
            {hylleplasser.map(h => (
              <option key={h} value={h}>{h}</option>
            ))}
          </select>

          <button
            className={`btn btn-secondary ${showRestockOnly ? 'active' : ''}`}
            onClick={() => setShowRestockOnly(!showRestockOnly)}
          >
            <ShoppingCart size={18} /> Trenger påfyll
          </button>

          <button className="btn" onClick={() => setShowAddForm(!showAddForm)}>
            <Plus size={18} /> Ny komponent
          </button>

          <button className="btn btn-secondary" onClick={loadComponents} disabled={loading}>
            {loading ? <Loader2 size={18} className="loading" /> : <RefreshCw size={18} />}
          </button>
        </div>
      </div>

      {/* Add Form */}
      {showAddForm && (
        <div className="add-form">
          <h3><Plus size={20} /> Legg til komponent</h3>
          <div className="form-grid">
            <div className="form-group">
              <label>Navn *</label>
              <input
                type="text"
                value={newComponent.name}
                onChange={(e) => setNewComponent(prev => ({ ...prev, name: e.target.value }))}
                className="input-field"
                placeholder="Komponentnavn"
              />
            </div>
            <div className="form-group">
              <label>Hylleplass *</label>
              <input
                type="text"
                value={newComponent.hylleplass}
                onChange={(e) => setNewComponent(prev => ({ ...prev, hylleplass: e.target.value }))}
                className="input-field"
                placeholder="F.eks. A1, B2"
                list="hylleplasser-list"
              />
              <datalist id="hylleplasser-list">
                {hylleplasser.map(h => <option key={h} value={h} />)}
              </datalist>
            </div>
            <div className="form-group">
              <label>Kategori</label>
              <select
                value={newComponent.kategori}
                onChange={(e) => setNewComponent(prev => ({ ...prev, kategori: e.target.value }))}
                className="select-field"
              >
                {kategorier.map(k => (
                  <option key={k} value={k}>{k}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label>Antall</label>
              <input
                type="number"
                value={newComponent.antall}
                onChange={(e) => setNewComponent(prev => ({ ...prev, antall: parseInt(e.target.value) || 0 }))}
                className="input-field"
                min="0"
              />
            </div>
            <div className="form-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={newComponent.forbruksvare}
                  onChange={(e) => setNewComponent(prev => ({ ...prev, forbruksvare: e.target.checked }))}
                />
                Forbruksvare
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={newComponent.restock}
                  onChange={(e) => setNewComponent(prev => ({ ...prev, restock: e.target.checked }))}
                />
                Trenger påfyll
              </label>
            </div>
          </div>
          <div className="form-actions">
            <button className="btn btn-success" onClick={handleAddComponent}>
              <Check size={18} /> Lagre
            </button>
            <button className="btn btn-secondary" onClick={() => setShowAddForm(false)}>
              <X size={18} /> Avbryt
            </button>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="message error">
          <AlertTriangle size={18} /> {error}
        </div>
      )}

      {/* Components Table */}
      <div className="table-container">
        {loading ? (
          <div className="loading-state">
            <Loader2 size={32} className="loading" />
            <p>Laster komponenter...</p>
          </div>
        ) : filteredComponents.length === 0 ? (
          <div className="empty-state">
            <Package size={48} />
            <p>Ingen komponenter funnet</p>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Navn</th>
                <th><MapPin size={14} /> Hylleplass</th>
                <th><Tag size={14} /> Kategori</th>
                <th>Antall</th>
                <th>Forbruksvare</th>
                <th>Status</th>
                <th>Handlinger</th>
              </tr>
            </thead>
            <tbody>
              {filteredComponents.map(component => (
                <tr key={component.id} className={component.restock ? 'needs-restock' : ''}>
                  {editingId === component.id ? (
                    <>
                      <td>
                        <input
                          type="text"
                          value={editForm.name}
                          onChange={(e) => setEditForm(prev => ({ ...prev, name: e.target.value }))}
                          className="input-field"
                        />
                      </td>
                      <td>
                        <input
                          type="text"
                          value={editForm.hylleplass}
                          onChange={(e) => setEditForm(prev => ({ ...prev, hylleplass: e.target.value }))}
                          className="input-field"
                          list="hylleplasser-edit"
                        />
                        <datalist id="hylleplasser-edit">
                          {hylleplasser.map(h => <option key={h} value={h} />)}
                        </datalist>
                      </td>
                      <td>
                        <select
                          value={editForm.kategori || 'Annet'}
                          onChange={(e) => setEditForm(prev => ({ ...prev, kategori: e.target.value }))}
                          className="select-field"
                          style={{ width: '120px', marginBottom: 0 }}
                        >
                          {kategorier.map(k => (
                            <option key={k} value={k}>{k}</option>
                          ))}
                        </select>
                      </td>
                      <td>
                        <input
                          type="number"
                          value={editForm.antall}
                          onChange={(e) => setEditForm(prev => ({ ...prev, antall: parseInt(e.target.value) || 0 }))}
                          className="input-field"
                          min="0"
                          style={{ width: '80px' }}
                        />
                      </td>
                      <td>
                        <label className="checkbox-inline">
                          <input
                            type="checkbox"
                            checked={editForm.forbruksvare}
                            onChange={(e) => setEditForm(prev => ({ ...prev, forbruksvare: e.target.checked }))}
                          />
                          Forbruk
                        </label>
                      </td>
                      <td>
                        <label className="checkbox-inline">
                          <input
                            type="checkbox"
                            checked={editForm.restock}
                            onChange={(e) => setEditForm(prev => ({ ...prev, restock: e.target.checked }))}
                          />
                          Påfyll
                        </label>
                      </td>
                      <td className="actions">
                        <button className="btn-icon success" onClick={saveEdit} title="Lagre">
                          <Check size={16} />
                        </button>
                        <button className="btn-icon" onClick={cancelEdit} title="Avbryt">
                          <X size={16} />
                        </button>
                      </td>
                    </>
                  ) : (
                    <>
                      <td className="name-cell">{component.name}</td>
                      <td>{component.hylleplass}</td>
                      <td>{component.kategori || 'Annet'}</td>
                      <td>{component.antall || '-'}</td>
                      <td>{component.forbruksvare ? 'Ja' : 'Nei'}</td>
                      <td>
                        <button
                          className={`restock-btn ${component.restock ? 'active' : ''}`}
                          onClick={() => toggleRestock(component)}
                          title={component.restock ? 'Fjern fra påfyllingsliste' : 'Merk for påfyll'}
                        >
                          <ShoppingCart size={14} />
                          {component.restock ? 'Trenger påfyll' : 'OK'}
                        </button>
                      </td>
                      <td className="actions">
                        <button className="btn-icon" onClick={() => startEdit(component)} title="Rediger">
                          <Edit3 size={16} />
                        </button>
                        <button className="btn-icon danger" onClick={() => handleDelete(component.id, component.name)} title="Slett">
                          <Trash2 size={16} />
                        </button>
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="table-footer">
        <span>{filteredComponents.length} komponenter</span>
      </div>
    </div>
  );
}

export default ComponentsManager;
