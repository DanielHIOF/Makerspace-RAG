/**
 * Admin API Service for Makerspace RAG
 */

const API_BASE = '';

export async function getStats() {
  const response = await fetch(`${API_BASE}/admin/stats`, {
    credentials: 'include'
  });
  if (!response.ok) {
    // Fallback to status endpoint
    const statusResp = await fetch(`${API_BASE}/status`);
    return statusResp.json();
  }
  return response.json();
}

export async function extractPDF(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/extract-pdf`, {
    method: 'POST',
    body: formData
  });

  return response.json();
}

export async function extractXLSX(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/extract-xlsx`, {
    method: 'POST',
    body: formData
  });

  return response.json();
}

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData
  });

  return response.json();
}

export async function enhancePDF(text, context, category) {
  const response = await fetch(`${API_BASE}/enhance-pdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, context, category })
  });

  return response.json();
}

export async function approveSummary(content, category) {
  const response = await fetch(`${API_BASE}/approve-summary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, category })
  });

  return response.json();
}

export async function approveXLSX(items) {
  const response = await fetch(`${API_BASE}/approve-xlsx`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items })
  });

  return response.json();
}

export async function addText(text) {
  const response = await fetch(`${API_BASE}/add-text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });

  return response.json();
}

export async function reloadIndex() {
  const response = await fetch(`${API_BASE}/reload`, {
    method: 'POST'
  });

  return response.json();
}

export default {
  getStats,
  extractPDF,
  extractXLSX,
  uploadFile,
  enhancePDF,
  approveSummary,
  approveXLSX,
  addText,
  reloadIndex
};
