import styled from 'styled-components';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PropTypes from 'prop-types';

const AssistantButton = styled.button`
  position: fixed;
  bottom: 24px;
  right: 24px;
  width: 60px;
  height: 60px;
  border-radius: 30px;
  background: ${props => props.theme.colors.primary};
  color: white;
  border: none;
  box-shadow: ${props => props.theme.shadows.medium};
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;

  &:hover {
    transform: scale(1.05);
    background: ${props => props.theme.colors.success};
  }
`;

const AIButton = ({ onClick }) => {
  return (
    <AssistantButton onClick={onClick}>
      <SmartToyIcon fontSize="large" />
    </AssistantButton>
  );
};
AIButton.propTypes = {
  onClick: PropTypes.func.isRequired
};

export default AIButton;
