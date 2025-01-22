import { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';
import { io } from 'socket.io-client';

const ChatContainer = styled.div`
  width: 600px;
  max-width: 90vw;
  height: 500px;
  background: white;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const ChatHeader = styled.div`
  padding: 16px;
  background: ${props => props.theme.colors.primary};
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;

  .timer {
    font-weight: 500;
  }
`;

const MessagesContainer = styled.div`
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const Message = styled.div`
  max-width: 80%;
  padding: 8px 12px;
  border-radius: 12px;
  align-self: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  background: ${props => props.isUser ? props.theme.colors.primary : props.theme.colors.background};
  color: ${props => props.isUser ? 'white' : 'inherit'};

  .sender {
    font-size: 12px;
    opacity: 0.8;
    margin-bottom: 4px;
  }
`;

const InputArea = styled.div`
  padding: 16px;
  border-top: 1px solid ${props => props.theme.colors.lightGray};
  display: flex;
  gap: 12px;

  input {
    flex: 1;
    padding: 8px 12px;
    border: 1px solid ${props => props.theme.colors.lightGray};
    border-radius: 20px;
    
    &:focus {
      outline: none;
      border-color: ${props => props.theme.colors.primary};
    }
  }

  button {
    padding: 8px 16px;
    background: ${props => props.theme.colors.primary};
    color: white;
    border: none;
    border-radius: 20px;
    cursor: pointer;

    &:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
  }
`;

const InterviewChat = ({ interview, onClose }) => {
  const [socket, setSocket] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [timeLeft, setTimeLeft] = useState(300);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    const newSocket = io(import.meta.env.VITE_WS_URL || 'http://localhost:8000', {
      query: { interviewId: interview.id }
    });

    newSocket.on('message', (message) => {
      setMessages(prev => [...prev, message]);
    });

    newSocket.on('interview_ended', () => {
      onClose();
    });

    setSocket(newSocket);

    return () => newSocket.close();
  }, [interview.id, onClose]);

  useEffect(() => {
    if (timeLeft > 0) {
      const timer = setInterval(() => setTimeLeft(t => t - 1), 1000);
      return () => clearInterval(timer);
    } else {
      socket?.emit('end_interview');
      onClose();
    }
  }, [timeLeft, onClose, socket]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = () => {
    if (!input.trim() || !socket) return;

    socket.emit('message', {
      text: input,
      sender: interview.userId,
      timestamp: new Date().toISOString()
    });
    setInput('');
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <ChatContainer onClick={e => e.stopPropagation()}>
      <ChatHeader>
        <div>Interview with {interview.freelancerName}</div>
        <div className="timer">{formatTime(timeLeft)}</div>
      </ChatHeader>

      <MessagesContainer>
        {messages.map((msg, i) => (
          <Message key={i} isUser={msg.sender === interview.userId}>
            <div className="sender">{msg.sender === interview.userId ? 'You' : interview.freelancerName}</div>
            <div>{msg.text}</div>
          </Message>
        ))}
        <div ref={messagesEndRef} />
      </MessagesContainer>

      <InputArea>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message..."
        />
        <button onClick={sendMessage}>Send</button>
      </InputArea>
    </ChatContainer>
  );
};

InterviewChat.propTypes = {
  interview: PropTypes.shape({
    id: PropTypes.string.isRequired,
    userId: PropTypes.string.isRequired,
    freelancerName: PropTypes.string.isRequired
  }).isRequired,
  onClose: PropTypes.func.isRequired
};

export default InterviewChat;
