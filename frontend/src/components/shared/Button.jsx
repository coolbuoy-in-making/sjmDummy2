import styled from 'styled-components';

export const Button = styled.button`
  padding: ${props => props.size === 'large' ? '16px 32px' : '12px 24px'};
  border-radius: 24px;
  font-size: ${props => props.size === 'large' ? '18px' : '16px'};
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  
  ${props => props.variant === 'primary' && `
    background: ${props.theme.colors.primary};
    color: white;
    border: none;
    
    &:hover {
      background: ${props.theme.colors.success};
    }
  `}
  
  ${props => props.variant === 'secondary' && `
    background: transparent;
    color: ${props.theme.colors.dark};
    border: 1px solid ${props.theme.colors.dark};
    
    &:hover {
      background: ${props.theme.colors.lightGray};
    }
  `}
`;

export default Button;