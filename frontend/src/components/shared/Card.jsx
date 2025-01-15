import PropTypes from 'prop-types';
import styled from 'styled-components';

const StyledCard = styled.div`
  background: white;
  border-radius: ${props => props.radius || '16px'};
  padding: ${props => props.padding || '24px'};
  border: 1px solid ${props => props.border || '#e4ebe4'};
  box-shadow: ${props => props.elevation === 'high' 
    ? '0 4px 12px rgba(0, 0, 0, 0.1)' 
    : props.elevation === 'medium'
    ? '0 2px 8px rgba(0, 0, 0, 0.05)'
    : 'none'};
  transition: all 0.2s ease;

  ${props => props.hoverable && `
    cursor: pointer;
    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
  `}

  ${props => props.variant === 'outlined' && `
    background: transparent;
    border: 1px solid #e4ebe4;
    box-shadow: none;
  `}

  ${props => props.variant === 'success' && `
    background: #f3fdf3;
    border: 1px solid #d3f0d3;
  `}

  ${props => props.variant === 'warning' && `
    background: #fff8e6;
    border: 1px solid #ffe0b2;
  `}
`;

/**
 * Card component that supports various display styles and interactions
 * @param {Object} props
 * @param {React.ReactNode} props.children - Child elements to render inside the card
 * @param {string} [props.className] - Additional CSS classes
 * @param {'default'|'outlined'|'success'|'warning'} [props.variant] - Visual style variant
 * @param {'none'|'medium'|'high'} [props.elevation] - Shadow elevation level
 * @param {boolean} [props.hoverable] - Enable hover animation
 * @param {string} [props.padding] - Custom padding value
 * @param {string} [props.radius] - Custom border radius
 * @param {function} [props.onClick] - Click handler
 */
const Card = ({ 
  children, 
  className,
  variant,
  elevation,
  hoverable,
  padding,
  radius,
  onClick
}) => {
  return (
    <StyledCard 
      className={className}
      variant={variant}
      elevation={elevation}
      hoverable={hoverable}
      padding={padding}
      radius={radius}
      onClick={onClick}
    >
      {children}
    </StyledCard>
  );
};

Card.propTypes = {
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  variant: PropTypes.oneOf(['default', 'outlined', 'success', 'warning']),
  elevation: PropTypes.oneOf(['none', 'medium', 'high']),
  hoverable: PropTypes.bool,
  padding: PropTypes.string,
  radius: PropTypes.string,
  onClick: PropTypes.func
};

Card.defaultProps = {
  variant: 'default',
  elevation: 'none',
  hoverable: false,
  padding: '24px',
  radius: '16px'
};

export default Card;
