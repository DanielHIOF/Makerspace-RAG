/**
 * API Service for Makerspace RAG
 */

const API_BASE = '';

export async function sendChatMessage(message, history = [], summary = '') {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      message,
      history,
      summary
    })
  });

  if (!response.ok) {
    throw new Error('Chat request failed');
  }

  return response.json();
}

export async function getStatus() {
  const response = await fetch(`${API_BASE}/status`);
  return response.json();
}

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}

export default {
  sendChatMessage,
  getStatus,
  getHealth
};
