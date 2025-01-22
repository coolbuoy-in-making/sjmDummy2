import { useState, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';

const Card = styled.div`
  padding: 20px;
  background: white;
  border-radius: 12px;
  box-shadow: ${props => props.theme.shadows.medium};
  margin: 16px 0;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;

  h4 {
    margin: 0;
    color: ${props => props.theme.colors.primary};
  }
`;

const Timer = styled.div`
  font-size: 18px;
  font-weight: 500;
  color: ${props => props.timeLeft <= 60 ? 'red' : props.theme.colors.text};
`;

const Question = styled.div`
  margin: 16px 0;
  
  .question-text {
    font-weight: 500;
    margin-bottom: 8px;
  }
  
  textarea {
    width: 100%;
    padding: 12px;
    border: 1px solid ${props => props.theme.colors.lightGray};
    border-radius: 8px;
    min-height: 100px;
    resize: vertical;
    
    &:focus {
      outline: none;
      border-color: ${props => props.theme.colors.primary};
    }
  }
`;

const ActionButton = styled.button`
  padding: 8px 16px;
  background: ${props => props.variant === 'primary' ? props.theme.colors.primary : 'white'};
  color: ${props => props.variant === 'primary' ? 'white' : props.theme.colors.primary};
  border: 1px solid ${props => props.theme.colors.primary};
  border-radius: 8px;
  cursor: pointer;
  margin-left: 8px;
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const InterviewCard = ({ interview, onComplete, onClose }) => {
  const [timeLeft, setTimeLeft] = useState(300); // 5 minutes in seconds
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({});
  
  const questions = [
    "How would you approach this project?",
    "What similar projects have you worked on?",
    "How do you handle project deadlines?",
    "What's your communication style with clients?",
    "Do you have any questions about the project?"
  ];

  useEffect(() => {
    if (timeLeft > 0) {
      const timer = setInterval(() => setTimeLeft(t => t - 1), 1000);
      return () => clearInterval(timer);
    } else {
      handleSubmit();
    }
  }, [timeLeft, handleSubmit]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleAnswer = (answer) => {
    setAnswers(prev => ({
      ...prev,
      [currentQuestion]: answer
    }));
  };

  const handleNext = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(c => c + 1);
    }
  };

  const handleSubmit = useCallback(() => {
    onComplete({
      interviewId: interview.id,
      answers,
      timeSpent: 300 - timeLeft
    });
  }, [interview.id, answers, timeLeft, onComplete]);

  return (
    <Card>
      <Header>
        <h4>Quick Interview</h4>
        <Timer timeLeft={timeLeft}>{formatTime(timeLeft)}</Timer>
      </Header>
      
      <Question>
        <div className="question-text">{questions[currentQuestion]}</div>
        <textarea
          value={answers[currentQuestion] || ''}
          onChange={(e) => handleAnswer(e.target.value)}
          placeholder="Type your answer here..."
        />
      </Question>

      <div style={{ textAlign: 'right' }}>
        <ActionButton onClick={onClose}>Cancel</ActionButton>
        {currentQuestion < questions.length - 1 ? (
          <ActionButton 
            variant="primary"
            onClick={handleNext}
            disabled={!answers[currentQuestion]}
          >
            Next
          </ActionButton>
        ) : (
          <ActionButton 
            variant="primary"
            onClick={handleSubmit}
            disabled={!answers[currentQuestion]}
          >
            Submit
          </ActionButton>
        )}
      </div>
    </Card>
  );
};

InterviewCard.propTypes = {
  interview: PropTypes.object.isRequired,
  onComplete: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired
};

export default InterviewCard;
