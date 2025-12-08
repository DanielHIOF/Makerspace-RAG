import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from './hooks/useTheme'
import { ChatProvider } from './hooks/useChat'
import ChatPage from './pages/ChatPage'
import AdminPage from './pages/AdminPage'

function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <Routes>
          <Route path="/" element={
            <ChatProvider>
              <ChatPage />
            </ChatProvider>
          } />
          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </ThemeProvider>
    </BrowserRouter>
  )
}

export default App
