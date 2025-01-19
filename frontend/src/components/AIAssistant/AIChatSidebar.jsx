import { useState, useRef, useEffect, useContext } from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import { UserContext } from '../../contexts/UserContext';

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

const SendButton = styled.button`
  background: ${props => props.theme.colors.primary};
  color: white;
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  
  &:hover {
    opacity: 0.9;
  }
`;

const LoadingMessage = styled.div`
display: flex;
align-items: center;
gap: 8px;
padding: 8px;
color: ${props => props.theme.colors.primary};
font-style: italic;
`;

const Spinner = styled.div`
width: 20px;
height: 20px;
border: 2px solid ${props => props.theme.colors.primary};
border-top-color: transparent;
border-radius: 50%;
animation: spin 1s linear infinite;

@keyframes spin {
  to { transform: rotate(360deg); }
}
`;

const Title = styled.h3`
  margin: 0;
  color: ${props => props.theme.colors.primary};
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  cursor: pointer;
  color: ${props => props.theme.colors.gray};
  
  &:hover {
    color: ${props => props.theme.colors.primary};
  }
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

const MessageWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: ${props => props.isUser ? 'flex-end' : 'flex-start'};
`;

const MessageText = styled.div`
  padding: 8px 12px;
  border-radius: 8px;
  background: ${props => props.isUser ? props.theme.colors.primary : props.theme.colors.background};
  color: ${props => props.isUser ? 'white' : props.theme.colors.text};
  white-space: pre-wrap;
`;

const FreelancerListContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const SuggestionsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const SuggestionButton = styled.button`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.lightGray};
  border-radius: 16px;
  padding: 8px 16px;
  cursor: pointer;
  
  &:hover {
    background: ${props => props.theme.colors.primary};
    color: white;
  }
`;

const MatchStats = styled.div`
  background: ${props => props.theme.colors.background};
  padding: 12px;
  border-radius: 8px;
  margin-bottom: 12px;
  
  .stat-item {
    display: flex;
    justify-content: space-between;
    margin: 4px 0;
  }
`;

const SkillTag = styled.span`
  background: ${props => props.matched ? props.theme.colors.primary : props.theme.colors.background};
  color: ${props => props.matched ? 'white' : props.theme.colors.text};
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  margin: 2px;
`;

const NoMatchMessage = styled.div`
  text-align: center;
  padding: 20px;
  background: ${props => props.theme.colors.background};
  border-radius: 8px;
  margin: 12px 0;
`;

const DebugInfo = styled.div`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.lightGray};
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
  font-size: 12px;
  font-family: monospace;
  white-space: pre-wrap;
  
  .debug-title {
    font-weight: bold;
    color: ${props => props.theme.colors.primary};
  }
`;

const ProcessStep = styled.div`
  color: ${props => props.success ? 'green' : props.error ? 'red' : props.theme.colors.primary};
  padding: 4px 0;
  font-style: italic;
`;

const DebugPanel = styled(DebugInfo)`
  .match-details {
    margin-top: 8px;
    padding-left: 12px;
    border-left: 2px solid ${props => props.theme.colors.primary};
  }
  
  .stat-value {
    color: ${props => props.theme.colors.primary};
    font-weight: bold;
  }
`;

const AIChatSidebar = ({ isOpen, onClose }) => {
  const { user } = useContext(UserContext);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [conversations, setConversations] = useState([]);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const [debugInfo, setDebugInfo] = useState(null);
  const [processingSteps, setProcessingSteps] = useState([]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addProcessingStep = (step, type = 'info') => {
    setProcessingSteps(prev => [...prev, { step, type, timestamp: new Date() }]);
  };

  const handleMessage = async (input) => {
    if (!input.trim()) return;

    setError(null);
    setLoading(true);
    setMessages(prev => [...prev, { text: input, isUser: true }]);
    setInput('');
    setDebugInfo(null);
    setProcessingSteps([]);
    
    addProcessingStep('Starting request processing...');

    try {
      if (input.toLowerCase().includes('find') || input.toLowerCase().includes('need')) {
        addProcessingStep('Detected search intent');
        addProcessingStep('Extracting skills and requirements...');
      }

      const response = await fetch(`${import.meta.env.VITE_PYTHON_URL || 'http://localhost:8000'}/ai/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          userType: user?.userType || 'client',
          userId: user?.id,
          conversationId: conversations.length > 0 ? conversations[0].id : null
        })
      });

      const data = await response.json();
      console.log('AI Response:', data); // Debug log

      if (!response.ok) {
        throw new Error(data.error || 'An error occurred');
      }

      if (data.success) {
        // Add debug information if available
        if (data.response.projectDetails) {
          addProcessingStep('Project details extracted successfully', 'success');
          setDebugInfo({
            extractedSkills: data.response.projectDetails.required_skills,
            budget: data.response.projectDetails.budget_range,
            complexity: data.response.projectDetails.complexity,
            timeline: data.response.projectDetails.timeline
          });
        }

        if (data.response.type === 'freelancerList') {
          addProcessingStep(`Found ${data.response.freelancers.length} matching freelancers`, 'success');
        } else if (data.response.type === 'suggestions') {
          addProcessingStep('No exact matches, generating suggestions', 'warning');
        }

        const newMessage = {
          id: Date.now(),
          timestamp: new Date().toISOString(),
          ...data.response
        };

        setConversations(prev => [newMessage, ...prev]);
        setMessages(prev => [...prev, newMessage]);
      }
    } catch (error) {
      const errorMessage = error.message || "Sorry, I'm having trouble connecting right now.";
      addProcessingStep(`Error: ${errorMessage}`, 'error');
      setError(errorMessage);
      setMessages(prev => [...prev, {
        text: errorMessage,
        isUser: false
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleInterviewRequest = async (freelancerId) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/ai/interview', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          freelancerId,
          userId: user?.id
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setMessages(prev => [...prev, {
          text: `Interview invitation sent to freelancer! I've prepared some relevant questions based on your project needs.`,
          isUser: false,
          type: 'interview',
          questions: data.questions
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        text: `Error: ${error.message || "Sorry, I couldn't send the interview invitation. Please try again."}`,
        isUser: false
      }]);
    } finally {
      setLoading(false);
    }
  };

  const renderFreelancerCard = (freelancer) => (
    <FreelancerCard key={freelancer.id}>
      <h5>{freelancer.name}</h5>
      <div>{freelancer.jobTitle}</div>
      <div className="skills">
        {freelancer.skills.map((skill, j) => (
          <SkillTag 
            key={j} 
            matched={freelancer.matchDetails?.skillMatch?.skills?.includes(skill)}
          >
            {skill}
          </SkillTag>
        ))}
      </div>
      <div className="stats">
        <span>${freelancer.hourlyRate}/hr</span>
        <span>{freelancer.rating}% success</span>
        <span>{freelancer.matchDetails.matchPercentage}% match</span>
      </div>
      {freelancer.matchDetails?.availability?.immediateStart && (
        <div className="availability">Available for immediate start</div>
      )}
      <button 
        onClick={() => handleInterviewRequest(freelancer.id)}
        disabled={!freelancer.matchDetails?.availability?.immediateStart}
      >
        Request Interview
      </button>
    </FreelancerCard>
  );

  const renderMessageDebugInfo = (message) => {
    if (!message.debugInfo) return null;
    
    const { searchCriteria, matchingProcess } = message.debugInfo;
    
    return (
      <DebugPanel>
        <div className="debug-title">AI Matching Process:</div>
        <div className="match-details">
          <div>Total Freelancers Searched: <span className="stat-value">{matchingProcess.totalFreelancers}</span></div>
          <div>Matches Found: <span className="stat-value">{matchingProcess.matchesFound}</span></div>
          <div>Search Criteria:</div>
          <div className="match-details">
            <div>Skills: {searchCriteria.required_skills.join(', ')}</div>
            <div>Budget: ${searchCriteria.budget_range[0]} - ${searchCriteria.budget_range[1]}</div>
            <div>Complexity: {searchCriteria.complexity}</div>
          </div>
        </div>
      </DebugPanel>
    );
  };

  const renderFreelancerList = (message) => (
    <>
      {renderMessageDebugInfo(message)}
      <FreelancerListContainer>
        <MessageText>{message.text}</MessageText>
        
        {message.matchStats && (
          <MatchStats>
            <div className="stat-item">
              <span>Total Matches:</span>
              <span>{message.matchStats.total}</span>
            </div>
            <div className="stat-item">
              <span>Highly Matched:</span>
              <span>{message.matchStats.highlyMatched}</span>
            </div>
            <div className="stat-item">
              <span>Skills Matched:</span>
              <span>{message.matchStats.skillsMatched}</span>
            </div>
          </MatchStats>
        )}
        
        {message.freelancers?.length > 0 ? (
          message.freelancers.map(freelancer => renderFreelancerCard(freelancer))
        ) : (
          <NoMatchMessage>
            <p>No freelancers found matching your exact criteria.</p>
            <p>Try broadening your search or consider different skill combinations.</p>
          </NoMatchMessage>
        )}
      </FreelancerListContainer>
    </>
  );

  const renderMessage = (message) => {
    switch (message.type) {
      case 'freelancerList':
        return renderFreelancerList(message);

      case 'suggestions':
        return (
          <SuggestionsContainer>
            <MessageText>{message.text}</MessageText>
            {message.suggestions?.map((suggestion, index) => (
              <SuggestionButton
                key={index}
                onClick={() => handleMessage(suggestion)}
              >
                {suggestion}
              </SuggestionButton>
            ))}
          </SuggestionsContainer>
        );

      default:
        return <MessageText isUser={message.isUser}>{message.text}</MessageText>;
    }
  };

  const renderDebugInfo = () => {
    if (!debugInfo) return null;
    
    return (
      <DebugInfo>
        <div className="debug-title">AI Processing Details:</div>
        <div>Extracted Skills: {debugInfo.extractedSkills.join(', ')}</div>
        <div>Budget Range: ${debugInfo.budget[0]} - ${debugInfo.budget[1]}</div>
        <div>Complexity: {debugInfo.complexity}</div>
        <div>Timeline: {debugInfo.timeline} days</div>
      </DebugInfo>
    );
  };

  useEffect(() => {
    // Load previous conversations from localStorage
    const savedConversations = localStorage.getItem(`chat_history_${user?.id}`);
    if (savedConversations) {
      setConversations(JSON.parse(savedConversations));
    }
  }, [user?.id]);

  useEffect(() => {
    // Save conversations to localStorage
    if (user?.id && conversations.length > 0) {
      localStorage.setItem(`chat_history_${user?.id}`, JSON.stringify(conversations));
    }
  }, [conversations, user?.id]);

  return (
    <Sidebar isOpen={isOpen}>
      <Header>
        <Title>AI Assistant</Title>
        <CloseButton onClick={onClose}><CloseIcon /></CloseButton>
      </Header>

      <MessagesContainer>
        {error && (
          <MessageText isUser={false} style={{ color: 'red' }}>
            {error}
          </MessageText>
        )}
        
        {processingSteps.map((step, index) => (
          <ProcessStep 
            key={index}
            success={step.type === 'success'}
            error={step.type === 'error'}
            warning={step.type === 'warning'}
          >
            {step.step}
          </ProcessStep>
        ))}

        {debugInfo && renderDebugInfo()}

        {messages.map((msg, idx) => (
          <MessageWrapper key={idx} isUser={msg.isUser}>
            {renderMessage(msg)}
          </MessageWrapper>
        ))}
        
        {loading && (
          <LoadingMessage>
            <Spinner />
            Processing...
          </LoadingMessage>
        )}
        <div ref={messagesEndRef} />
      </MessagesContainer>

      <InputContainer>
        <Input
          placeholder="Ask me to find freelancers or help with your project..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && !loading && handleMessage(e.target.value)}
          disabled={loading}
        />
        <SendButton 
          onClick={() => !loading && handleMessage(input)} 
          disabled={loading || !input.trim()}
        >
          <SendIcon />
        </SendButton>
      </InputContainer>
    </Sidebar>
  );
};

AIChatSidebar.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired
};

export default AIChatSidebar;
