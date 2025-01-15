import styled from 'styled-components';

const Input = styled.input`
  width: 100%;
  padding: 12px;
  border: 1px solid ${props => props.theme.colors.lightGray};
  border-radius: 8px;
  font-size: 16px;
  
  &:focus {
    border-color: ${props => props.theme.colors.primary};
    outline: none;
  }
  
  &::placeholder {
    color: ${props => props.theme.colors.gray};
  }
`;

export default Input;