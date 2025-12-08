import React from 'react'
import Header from '../components/Header'
import ChatArea from '../components/ChatArea'
import InputArea from '../components/InputArea'

function ChatPage() {
  return (
    <div className="chat-page">
      <Header />
      <ChatArea />
      <InputArea />
    </div>
  )
}

export default ChatPage
