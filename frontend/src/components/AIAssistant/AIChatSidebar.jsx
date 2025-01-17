import { useState, useRef, useEffect, useContext } from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import { UserContext } from '../../contexts/UserContext'; // Add this import

const Sidebar = styled.div`
  position: fixed;
  top: 0;
  right: ${props => props.isOpen ? '0' : '-400px'};
  width: 400px;
  height: 100vh;
  background: white;
  box-shadow: ${props => props.theme.shadows.large};
  transition: right 0.3s ease;
  z-index: 1000;
  display: flex;
  flex-direction: column;

  @media (max-width: ${props => props.theme.breakpoints.sm}) {
    width: 100%;
    right: ${props => props.isOpen ? '0' : '-100%'};
  }
`;

const Header = styled.div`
  padding: 16px;
  border-bottom: 1px solid ${props => props.theme.colors.lightGray};
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ChatContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 16px;
`;

const InputContainer = styled.div`
  padding: 16px;
  border-top: 1px solid ${props => props.theme.colors.lightGray};
  display: flex;
  gap: 8px;
`;

const Input = styled.input`
  flex: 1;
  padding: 12px;
  border: 1px solid ${props => props.theme.colors.lightGray};
  border-radius: 24px;
  font-size: 14px;

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const Message = styled.div`
  margin-bottom: 16px;
  padding: 12px;
  border-radius: 12px;
  max-width: 80%;
  
  ${props => props.isUser ? `
    background: ${props.theme.colors.primary};
    color: white;
    margin-left: auto;
  ` : `
    background: ${props.theme.colors.background};
    margin-right: auto;
  `}
`;

const FreelancerCard = styled.div`
  padding: 12px;
  border: 1px solid ${props => props.theme.colors.lightGray};
  border-radius: 8px;
  margin-bottom: 8px;

  h5 {
    margin: 0 0 8px 0;
    color: ${props => props.theme.colors.primary};
  }

  .skills {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin: 4px 0;
  }

  .skill-tag {
    background: ${props => props.theme.colors.background};
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
  }

  .stats {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-size: 14px;
  }
`;

const AIChatSidebar = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const { user } = useContext(UserContext);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);


  const API_URL = import.meta.env.VITE_PYTHON_URL || 'http://localhost:8000';
  const sendMessage = async (input) => {
    setMessages(prev => [...prev, { text: input, isUser: true }]);
    setInput('');

    try {
      const response = await fetch(`${API_URL}/ai/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          userType: user?.userType || 'client',
          userId: user?.id
        }),
      });

      const data = await response.json();
      
      // Handle different response types
      if (data.success) {
        if (data.response.freelancers) {
          // Show matched freelancers
          setMessages(prev => [...prev, {
            type: 'freelancerList',
            freelancers: data.response.freelancers,
            isUser: false
          }]);
        } else if (data.response.questions) {
          // Show interview questions
          setMessages(prev => [...prev, {
            type: 'interview',
            questions: data.response.questions,
            isUser: false
          }]);
        } else {
          // Show regular message
          setMessages(prev => [...prev, {
            text: data.response.text,
            isUser: false
          }]);
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        text: "Sorry, I'm having trouble connecting right now.",
        isUser: false
      }]);
    }
  };

  const handleSendMessage = () => {
    if (!input.trim()) return;
    sendMessage(input);
  };

  const renderFreelancerList = (freelancers) => (
    <div>
      <h4>Matched Freelancers:</h4>
      {freelancers.map((freelancer, i) => (
        <FreelancerCard key={freelancer.id || i}>
          <h5>{freelancer.name}</h5>
          <div>{freelancer.jobTitle}</div>
          <div className="skills">
            {freelancer.skills.map((skill, j) => (
              <span key={j} className="skill-tag">{skill}</span>
            ))}
          </div>
          <div className="stats">
            <span>${freelancer.hourlyRate}/hr</span>
            <span>{freelancer.rating}% success</span>
            <span>{Math.round(freelancer.score * 100)}% match</span>
          </div>
          {freelancer.profileUrl && (
            <a href={freelancer.profileUrl} target="_blank" rel="noopener noreferrer">
              View Profile
            </a>
          )}
        </FreelancerCard>
      ))}
    </div>
  );

  const renderMessage = (msg, index) => {
    if (msg.type === 'freelancerList') {
      return (
        <Message key={index} isUser={msg.isUser}>
          {renderFreelancerList(msg.freelancers)}
        </Message>
      );
    }
    
    if (msg.type === 'interview') {
      return (
        <Message key={index} isUser={msg.isUser}>
          <h4>Interview Questions:</h4>
          {msg.questions.map((q, i) => (
            <div key={i}>{q}</div>
          ))}
        </Message>
      );
    }

    return (
      <Message key={index} isUser={msg.isUser}>
        {msg.text}
      </Message>
    );
  };

  return (
    <Sidebar isOpen={isOpen}>
      <Header>
        <h3>Upwork AI Assistant</h3>
        <CloseIcon 
          onClick={onClose}
          style={{ cursor: 'pointer' }}
        />
      </Header>
      
      <ChatContainer>
        {messages.map((msg, index) => renderMessage(msg, index))}
        <div ref={messagesEndRef} />
      </ChatContainer>

      <InputContainer>
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Ask me anything..."
        />
        <button 
          className="primary-button"
          onClick={handleSendMessage}
          style={{ padding: '12px' }}
        >
          <SendIcon />
        </button>
      </InputContainer>
    </Sidebar>
  );
};

AIChatSidebar.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired
};

export default AIChatSidebar;
