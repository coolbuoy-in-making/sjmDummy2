import { useState, useRef, useEffect, useContext } from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import { UserContext } from '../../contexts/UserContext';
import { useNavigate } from 'react-router-dom';
import api from '../../utils/api';

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
  padding: 20px;
  border: 1px solid ${props => props.theme.colors.lightGray};
  border-radius: 12px;
  background: white;
  transition: all 0.2s;

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.medium};
  }

  h5 {
    font-size: 18px;
    margin: 0 0 8px;
    color: ${props => props.theme.colors.primary};
  }

  .job-title {
    color: ${props => props.theme.colors.dark};
    font-size: 14px;
    margin-bottom: 12px;
  }

  .skills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0;
  }

  .stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 16px 0;
    font-size: 14px;
    color: ${props => props.theme.colors.dark};

    @media (max-width: 480px) {
      grid-template-columns: 1fr;
      text-align: center;
    }
  }

  @media (max-width: 768px) {
    padding: 16px;
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

const LoadingOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1001;
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
  -webkit-overflow-scrolling: touch; // Smooth scrolling on iOS

  @media (max-width: 768px) {
    padding: 12px;
    gap: 12px;
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
  background: ${props => props.matched 
    ? `${props.theme.colors.primary}20` 
    : props.theme.colors.background};
  color: ${props => props.matched 
    ? props.theme.colors.primary 
    : props.theme.colors.text};
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 13px;
  border: 1px solid ${props => props.matched 
    ? props.theme.colors.primary 
    : 'transparent'};
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

// Add new styled components
const CardActions = styled.div`
  display: flex;
  gap: 12px;
  margin-top: 12px;
`;

const ActionButton = styled.button`
  flex: 1;
  padding: 8px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
  
  &.primary {
    background: ${props => props.theme.colors.primary};
    color: white;
    
    &:hover {
      opacity: 0.9;
    }
  }
  
  &.secondary {
    background: white;
    color: ${props => props.theme.colors.primary};
    border: 1px solid ${props => props.theme.colors.primary};
    
    &:hover {
      background: ${props => props.theme.colors.background};
    }
  }

  &:disabled {
    background: ${props => props.theme.colors.lightGray};
    border-color: transparent;
    cursor: not-allowed;
  }
`;

const StatusIndicator = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: ${props => props.online ? 'green' : props.theme.colors.gray};
  font-size: 12px;
  
  &::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: ${props => props.online ? 'green' : props.theme.colors.lightGray};
  }
`;

const SkillList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 8px 0;
`;

const SkillPill = styled.span`
  background: ${props => props.theme.colors.background};
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 4px;

  button {
    background: none;
    border: none;
    color: ${props => props.theme.colors.gray};
    cursor: pointer;
    padding: 0;
    font-size: 16px;
    line-height: 1;
    
    &:hover {
      color: ${props => props.theme.colors.primary};
    }
  }
`;
const AIChatSidebar = ({ isOpen, onClose }) => {
  const navigate = useNavigate();
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
    
    // Debug logging
    console.log('Sending request:', { input, details });

    try {
      const url = `${import.meta.env.VITE_PYTHON_URL || 'http://localhost:8000'}/ai/chat`;
      console.log('Request URL:', url);

      const requestBody = {
        message: input,
        userType: user?.userType || 'client',
        userId: user?.id,
        projectDetails: details
      };
      console.log('Request body:', requestBody);

      const response = await fetch(url, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      const data = await response.json();
      console.log('Raw AI Response:', data);

      if (!response.ok) {
        throw new Error(data.error || 'An error occurred');
      }

      if (data.success && data.response) {
        // Transform freelancers data if present
        if (data.response.type === 'freelancerList' && data.response.freelancers) {
          data.response.freelancers = data.response.freelancers.map(f => ({
            ...f,
            skills: Array.isArray(f.skills) ? f.skills : [],
            matchDetails: {
              skillMatch: f.matchDetails?.skillMatch || { skills: [], count: 0 },
              matchPercentage: f.matchDetails?.matchPercentage || 0
            }
          }));
        }

        const aiMessage = {
          id: Date.now(),
          timestamp: new Date().toISOString(),
          isUser: false,
          ...data.response
        };

        console.log('Processed message:', aiMessage);
        setMessages(prev => [...prev, aiMessage]);
        setConversations(prev => [aiMessage, ...prev]);
      }
    } catch (error) {
      console.error('AI Chat Error:', error);
      setError(error.message || "Sorry, I'm having trouble connecting right now.");
    } finally {
      setLoading(false);
    }
  };

  // Update the handleInterviewRequest function
  const handleInterviewRequest = async (freelancer) => {
    setLoading(true);
    try {
      if (!user) {
        navigate('/login', { 
          state: { 
            returnTo: window.location.pathname,
            message: 'Please log in to request interviews' 
          }
        });
        return;
      }
  
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('Please log in to continue');
      }
  
      const response = await api.post('/freelancers/interview', {
        freelancerId: freelancer.id,
        projectId: projectDetails?.id,
        message: `Interview request for ${projectDetails?.skills?.join(', ')} project`
      });
  
      if (response.data.success) {
        setMessages(prev => [...prev, {
          text: `Interview request sent to ${freelancer.name}! They will be notified and can respond to your request.`,
          isUser: false,
          type: 'success'
        }]);
      }
    } catch (error) {
      if (error.response?.status === 401) {
        // Handle unauthorized - likely token expired
        setMessages(prev => [...prev, {
          text: "Your session has expired. Please log in again.",
          isUser: false,
          type: 'error'
        }]);
        
        // Delay navigation to show the message
        setTimeout(() => {
          navigate('/login', { 
            state: { 
              returnTo: window.location.pathname,
              message: 'Please log in again to continue' 
            }
          });
        }, 2000);
      } else {
        setMessages(prev => [...prev, {
          text: error.response?.data?.message || "Failed to send interview request. Please try again.",
          isUser: false,
          type: 'error'
        }]);
      }
    } finally {
      setLoading(false);
    }
  };
  

  // Update the renderFreelancerCard function
  const renderFreelancerCard = (freelancer) => {
    if (!freelancer?.id) {
      console.log('Invalid freelancer data:', freelancer);
      return null;
    }
  
    // Ensure all required properties with defaults
    const safeFreelancer = {
      id: freelancer.id,
      name: freelancer.name || 'Unknown',
      jobTitle: freelancer.jobTitle || '',
      skills: Array.isArray(freelancer.skills) ? freelancer.skills : [],
      hourlyRate: parseFloat(freelancer.hourlyRate) || 0,
      rating: parseFloat(freelancer.rating) || 0,
      availability: Boolean(freelancer.availability),
      totalSales: parseInt(freelancer.totalSales) || 0,
      matchDetails: {
        skillMatch: {
          skills: [],
          count: 0,
          ...freelancer.matchDetails?.skillMatch
        },
        matchPercentage: freelancer.matchDetails?.matchPercentage || 0,
        experienceScore: freelancer.matchDetails?.experienceScore || 0,
        contentScore: freelancer.matchDetails?.contentScore || 0,
        collaborativeScore: freelancer.matchDetails?.collaborativeScore || 0
      }
    };
  
    return (
      <FreelancerCard key={safeFreelancer.id}>
        <div className="card-header">
          <h5>{safeFreelancer.name}</h5>
          <StatusIndicator online={safeFreelancer.availability}>
            {safeFreelancer.availability ? 'Available' : 'Busy'}
          </StatusIndicator>
        </div>
        
        <div className="job-title">{safeFreelancer.jobTitle}</div>
        
        <div className="skills">
          {safeFreelancer.skills.map((skill, j) => (
            <SkillTag 
              key={j} 
              matched={safeFreelancer.matchDetails.skillMatch.skills.includes(skill)}
            >
              {skill}
            </SkillTag>
          ))}
        </div>
  
        <div className="stats">
          <div className="stat">
            <span className="label">Rate:</span>
            <span className="value">${safeFreelancer.hourlyRate}/hr</span>
          </div>
          <div className="stat">
            <span className="label">Success:</span>
            <span className="value">{safeFreelancer.rating}%</span>
          </div>
          <div className="stat">
            <span className="label">Match:</span>
            <span className="value">{safeFreelancer.matchDetails.matchPercentage}%</span>
          </div>
        </div>
  
        <CardActions>
          <ActionButton 
            className="secondary"
            onClick={() => {
              navigate(`/profile/${safeFreelancer.id}`);
              onClose();
            }}
          >
            View Profile
          </ActionButton>
          <ActionButton 
            className="primary"
            onClick={() => handleInterviewRequest(safeFreelancer)}
            disabled={!safeFreelancer.availability}
          >
            Request Interview
          </ActionButton>
        </CardActions>
      </FreelancerCard>
    );
  };
  

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
          {message.suggestions?.map((suggestion, index) => (
            <div key={index} style={{ marginTop: '12px' }}>
              <p><strong>{suggestion.message}</strong></p>
              <SuggestionButton onClick={() => handleMessage('refine_search', suggestion.action)}>
                {suggestion.action}
              </SuggestionButton>
            </div>
          ))}
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
              setProjectDetails(null);
            }}
          >
            Edit Requirements
          </button>
          <button 
            className="confirm" 
            onClick={() => handleMessage('confirm project details', details)}
          >
            Find Freelancers
          </button>
        </div>
      </ProjectSummary>
    );
  };

  const renderProjectDetailsForm = () => {    
    return (
      <ProjectDetailsForm>
        <div className="field">
          <label>Required Skills:</label>
          <SkillList>
            {skills.map((skill, index) => (
              <SkillPill key={index}>
                {skill}
                <button onClick={() => removeSkill(skill)}>&times;</button>
              </SkillPill>
            ))}
          </SkillList>
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
      {loading && (
        <LoadingOverlay>
          <Spinner />
        </LoadingOverlay>
      )}
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
        
        {loading && (          <LoadingMessage>            <Spinner />            Processing...          </LoadingMessage>        )}        <div ref={messagesEndRef} />      </MessagesContainer>      <InputContainer>        <Input          placeholder="Ask me to find freelancers or help with your project..."          value={input}          onChange={(e) => setInput(e.target.value)}
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
