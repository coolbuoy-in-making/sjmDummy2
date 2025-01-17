import { useState, useRef, useEffect, useContext } from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import { UserContext } from '../../contexts/UserContext';

const Sidebar = styled.div`
  position: fixed;
  top: 0;
  right: ${props => props.isOpen ? '0' : '-450px'};
  width: 450px;
  height: 100vh;
  background: white;
  box-shadow: ${props => props.theme.shadows.large};
  transition: right 0.3s ease;
  z-index: 1000;
  display: flex;
  flex-direction: column;

  @media (max-width: 768px) {
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
  background: white;
  display: flex;
  gap: 12px;
  position: sticky;
  bottom: 0;

  @media (max-width: 768px) {
    padding: 12px;
  }
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
  padding: 16px;
  border: 1px solid ${props => props.theme.colors.lightGray};
  border-radius: 12px;
  margin-bottom: 12px;
  transition: transform 0.2s, box-shadow 0.2s;

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.medium};
  }

  h5 {
    margin: 0 0 12px 0;
    color: ${props => props.theme.colors.primary};
    font-size: 1.1rem;
  }

  .skills {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
  }

  .skill-tag {
    background: ${props => props.theme.colors.background};
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 13px;
  }

  .stats {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 12px;
    font-size: 14px;
  }

  button {
    margin-top: 12px;
    width: 100%;
    padding: 8px;
    border-radius: 8px;
    border: none;
    background: ${props => props.theme.colors.primary};
    color: white;
    cursor: pointer;
    transition: opacity 0.2s;

    &:hover {
      opacity: 0.9;
    }

    &:disabled {
      background: ${props => props.theme.colors.lightGray};
      cursor: not-allowed;
    }
  }

  @media (max-width: 768px) {
    padding: 12px;
    
    .stats {
      flex-direction: column;
      gap: 8px;
    }
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
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  scroll-behavior: smooth;

  @media (max-width: 768px) {
    padding: 12px;
  }

  &::-webkit-scrollbar {
    width: 6px;
  }

  &::-webkit-scrollbar-track {
    background: ${props => props.theme.colors.background};
  }

  &::-webkit-scrollbar-thumb {
    background: ${props => props.theme.colors.lightGray};
    border-radius: 3px;
  }
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

const MatchingProcess = styled.div`
  background: ${props => props.theme.colors.background};
  border-left: 3px solid ${props => props.theme.colors.primary};
  padding: 12px;
  margin: 8px 0;
  font-size: 14px;

  .step {
    margin: 4px 0;
    color: ${props => props.theme.colors.text};
  }

  .stats {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid ${props => props.theme.colors.lightGray};
    
    .stat-item {
      color: ${props => props.theme.colors.primary};
      font-weight: bold;
    }
  }
`;

const ProjectDetailsForm = styled.div`
  padding: 20px;
  background: ${props => props.theme.colors.background};
  border-radius: 12px;
  margin: 16px 0;
  max-width: 100%;

  .field {
    margin-bottom: 20px;
    
    label {
      display: block;
      margin-bottom: 8px;
      color: ${props => props.theme.colors.primary};
      font-weight: 500;
    }

    input, select {
      width: 100%;
      padding: 10px;
      border: 1px solid ${props => props.theme.colors.lightGray};
      border-radius: 8px;
      font-size: 14px;

      &:focus {
        outline: none;
        border-color: ${props => props.theme.colors.primary};
      }
    }

    .hint {
      font-size: 12px;
      color: ${props => props.theme.colors.gray};
      margin-top: 6px;
    }
  }

  button {
    width: 100%;
    padding: 12px;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 500;
  }

  @media (max-width: 768px) {
    padding: 16px;
    
    .field {
      margin-bottom: 16px;
    }
  }
`;

const ProjectSummary = styled.div`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.primary};
  border-radius: 8px;
  padding: 16px;
  margin: 12px 0;

  h4 {
    color: ${props => props.theme.colors.primary};
    margin: 0 0 12px 0;
  }

  .summary-item {
    display: flex;
    margin: 8px 0;
    
    .label {
      font-weight: 500;
      min-width: 120px;
    }
  }

  .action-buttons {
    display: flex;
    gap: 12px;
    margin-top: 16px;
  }

  button {
    flex: 1;
    padding: 8px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    
    &.confirm {
      background: ${props => props.theme.colors.primary};
      color: white;
    }
    
    &.edit {
      background: ${props => props.theme.colors.background};
      border: 1px solid ${props => props.theme.colors.primary};
      color: ${props => props.theme.colors.primary};
    }
  }
`;

const AIChatSidebar = ({ isOpen, onClose }) => {
  const { user } = useContext(UserContext);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([{
    text: "Hi! I'm your AI assistant. I can help you find freelancers or answer questions about your project.",
    isUser: false,
    type: 'text'
  }]);
  const [loading, setLoading] = useState(false);
  const [conversations, setConversations] = useState([]);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const removeSkill = (skillToRemove) => {
    setSkills(prevSkills => prevSkills.filter(skill => skill !== skillToRemove));
  };

  const addSkill = (skill) => {
    if (!skills.includes(skill)) {
      setSkills(prevSkills => [...prevSkills, skill]);
    }
  };

  const [debugInfo] = useState(null);
  // Remove unused state since processing steps are handled directly in the message flow
  const [processingSteps] = useState([]);
  const [projectDetails, setProjectDetails] = useState(null);
  const [skills, setSkills] = useState([]);
  const [newSkill, setNewSkill] = useState('');
  const [showSummary, setShowSummary] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);


  const handleMessage = async (input, details = null) => {
    if (!input.trim() && !details) return;

    setError(null);
    setLoading(true);
    
    // Add user message to chat
    const userMessage = { text: input, isUser: true, type: 'text' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    
    try {
      const response = await fetch(`${import.meta.env.VITE_PYTHON_URL || 'http://localhost:8000'}/ai/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          userType: user?.userType || 'client',
          userId: user?.id,
          projectDetails: details
        })
      });

      const data = await response.json();
      console.log('AI Response:', data); // Debug log

      if (!response.ok) {
        throw new Error(data.error || 'An error occurred');
      }

      if (data.success) {
        const aiMessage = {
          id: Date.now(),
          timestamp: new Date().toISOString(),
          isUser: false,
          ...data.response
        };

        setMessages(prev => [...prev, aiMessage]);
        setConversations(prev => [aiMessage, ...prev]);
      }
    } catch (error) {
      console.error('AI Chat Error:', error);
      const errorMessage = {
        text: error.message || "Sorry, I'm having trouble connecting right now.",
        isUser: false,
        type: 'error'
      };
      setError(errorMessage.text);
      setMessages(prev => [...prev, errorMessage]);
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
    <FreelancerListContainer>
      <MessageText>{message.text}</MessageText>
      
      {message.matchingProcess && (
        <MatchingProcess>
          <div className="debug-title">Matching Process:</div>
          {message.matchingProcess.steps.map((step, index) => (
            <div key={index} className="step">{step}</div>
          ))}
          <div className="stats">
            <div className="stat-item">
              Total Freelancers Searched: {message.matchingProcess.searchStats.totalFreelancers}
            </div>
            <div className="stat-item">
              Matches Found: {message.matchingProcess.searchStats.matchesFound}
            </div>
            <div className="stat-item">
              High Quality Matches: {message.matchingProcess.searchStats.highMatches}
            </div>
          </div>
        </MatchingProcess>
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
  );

  const handleProjectDetailsSubmit = (details) => {
    const formattedDetails = {
      ...details,
      skills: skills.length > 0 ? skills : details.skills
    };
    
    setShowSummary(true);
    setMessages(prev => [...prev, {
      type: 'project_summary',
      projectDetails: formattedDetails,
      isUser: false,
      text: "Here's a summary of your project requirements:"
    }]);
  };

  const renderProjectSummary = (message) => {
    const details = message.projectDetails;
    return (
      <ProjectSummary>
        <h4>Project Requirements</h4>
        <div className="summary-item">
          <span className="label">Skills:</span>
          <span>{details.skills.join(', ')}</span>
        </div>
        <div className="summary-item">
          <span className="label">Budget:</span>
          <span>{details.budget}</span>
        </div>
        <div className="summary-item">
          <span className="label">Complexity:</span>
          <span>{details.complexity}</span>
        </div>
        <div className="summary-item">
          <span className="label">Timeline:</span>
          <span>{details.timeline} days</span>
        </div>
        <div className="action-buttons">
          <button 
            className="edit" 
            onClick={() => {
              setShowSummary(false);
              setCurrentProjectDetails(null);
            }}
          >
            Edit Requirements
          </button>
          <button 
            className="confirm" 
            onClick={() => handleMessage('confirm_project_details', details)}
          >
            Find Freelancers
          </button>
        </div>
      </ProjectSummary>
    );
  };

  const renderProjectDetailsForm = (message) => {
    const { requiredInputs } = message;
    
    return (
      <ProjectDetailsForm>
        <div className="field">
          <label>Required Skills:</label>
          <div className="skill-tags">
            {requiredInputs.skills.initial.map((skill, index) => (
              <SkillTag key={index}>
                {skill}
                <button onClick={() => removeSkill(skill)}>×</button>
              </SkillTag>
            ))}
          </div>
          <Input
            placeholder="Add more skills (press Enter)"
            value={newSkill}
            onChange={(e) => setNewSkill(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && newSkill.trim()) {
                addSkill(newSkill.trim());
                setNewSkill('');
              }
            }}
          />
          <div className="hint">Press Enter to add each skill</div>
        </div>

        <div className="field">
          <label>Budget Range:</label>
          <Input
            id="budget"
            placeholder="e.g., $30-100/hr"
          />
          <div className="hint">Format: $min-max/hr</div>
        </div>

        <div className="field">
          <label>Project Complexity:</label>
          <select id="complexity" defaultValue="medium">
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
          <div className="hint">Select based on project requirements</div>
        </div>

        <div className="field">
          <label>Timeline (days):</label>
          <Input
            id="timeline"
            type="number"
            min="1"
            placeholder="e.g., 30"
          />
          <div className="hint">Estimated project duration in days</div>
        </div>

        <SendButton onClick={() => handleProjectDetailsSubmit({
          skills: skills,
          budget: document.getElementById('budget').value,
          complexity: document.getElementById('complexity').value,
          timeline: document.getElementById('timeline').value
        })}>
          Find Matching Freelancers
        </SendButton>
      </ProjectDetailsForm>
    );
  };

  const renderMessage = (message) => {
    // Add null check and type validation
    if (!message) return null;

    // Ensure message has required properties
    const safeMessage = {
      type: 'text',
      text: '',
      ...message
    };

    switch (safeMessage.type) {
      case 'project_details_request': {
        const extractedDetails = safeMessage.requiredInputs?.skills?.initial || [];
        if (!projectDetails) {
          setProjectDetails({ skills: extractedDetails });
          setSkills(extractedDetails);
        }
        return (
          <>
            <MessageText isUser={false}>{safeMessage.text}</MessageText>
            {!showSummary && renderProjectDetailsForm(safeMessage)}
          </>
        );
      }

      case 'project_summary':
        return renderProjectSummary(safeMessage);
        
      case 'freelancerList':
        return renderFreelancerList(safeMessage);

      case 'suggestions':
        return (
          <SuggestionsContainer>
            <MessageText>{safeMessage.text}</MessageText>
            {safeMessage.suggestions?.map((suggestion, index) => (
              <SuggestionButton
                key={index}
                onClick={() => handleMessage(suggestion)}
              >
                {suggestion}
              </SuggestionButton>
            ))}
            {safeMessage.debugInfo && renderMessageDebugInfo(safeMessage)}
          </SuggestionsContainer>
        );

      default:
        return (
          <>
            <MessageText isUser={safeMessage.isUser}>{safeMessage.text}</MessageText>
            {safeMessage.debugInfo && renderMessageDebugInfo(safeMessage)}
          </>
        );
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
