import { useState } from 'react';
import styled from 'styled-components';
import ChatList from './ChatList';
import ChatWindow from './ChatWindow';

const ChatContainer = styled.div`
  display: grid;
  grid-template-columns: 320px 1fr;
  height: calc(100vh - 64px);
  background: white;
  border-top: 1px solid #e4ebe4;
`;

const ChatInterface = () => {
  const [selectedChat, setSelectedChat] = useState(null);

  return (
    <ChatContainer>
      <ChatList onSelectChat={setSelectedChat} selectedChat={selectedChat} />
      <ChatWindow currentChat={selectedChat} />
    </ChatContainer>
  );
};

export default ChatInterface;
