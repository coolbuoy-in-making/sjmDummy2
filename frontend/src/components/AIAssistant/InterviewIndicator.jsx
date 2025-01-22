
import styled from 'styled-components';
import PropTypes from 'prop-types';

const StyledIndicator = styled.button`
  position: fixed;
  bottom: 80px;
  right: ${props => props.isSidebarOpen ? '460px' : '20px'};
  background: ${props => props.status === 'accepted' ? '#4CAF50' : props.theme.colors.primary};
  color: white;
  border: none;
  border-radius: 24px;
  padding: 8px 16px;
  font-size: 14px;
  cursor: ${props => props.status === 'accepted' ? 'pointer' : 'default'};
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.3s ease;
  z-index: 1000;
  box-shadow: ${props => props.theme.shadows.medium};
  opacity: ${props => props.status === 'accepted' ? 1 : 0.8};

  &:hover {
    transform: ${props => props.status === 'accepted' ? 'translateY(-2px)' : 'none'};
  }

  @media (max-width: 768px) {
    right: 20px;
    bottom: 20px;
  }
`;

const InterviewIndicator = ({ interview, isSidebarOpen, onClick }) => {
  if (!interview) return null;

  return (
    <StyledIndicator
      status={interview.status}
      isSidebarOpen={isSidebarOpen}
      onClick={() => interview.status === 'accepted' && onClick()}
    >
      {interview.status === 'accepted' 
        ? `✓ Start Interview with ${interview.freelancerName}`
        : 'Interview Request Pending...'}
    </StyledIndicator>
  );
};

InterviewIndicator.propTypes = {
  interview: PropTypes.shape({
    status: PropTypes.string.isRequired,
    freelancerName: PropTypes.string
  }),
  isSidebarOpen: PropTypes.bool.isRequired,
  onClick: PropTypes.func.isRequired
};

export default InterviewIndicator;