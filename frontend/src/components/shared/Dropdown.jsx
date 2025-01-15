import { useState } from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import KeyboardArrowDown from '@mui/icons-material/KeyboardArrowDown';

const DropdownContainer = styled.div`
  position: relative;
  display: inline-block;
`;

const DropdownButton = styled.button`
  background: none;
  border: none;
  padding: 8px 12px;
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  color: ${props => props.theme.colors.dark};
  font-weight: 500;
`;

const DropdownContent = styled.div`
  position: absolute;
  top: 100%;
  left: 0;
  min-width: 220px;
  background: white;
  border-radius: 8px;
  box-shadow: ${props => props.theme.shadows.medium};
  padding: 8px 0;
  z-index: 1000;
  display: ${props => props.isOpen ? 'block' : 'none'};
`;

const DropdownItem = styled.a`
  display: block;
  padding: 8px 16px;
  color: ${props => props.theme.colors.dark};
  text-decoration: none;
  
  &:hover {
    background: ${props => props.theme.colors.background};
  }
`;

export const NavDropdown = ({ title, items }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <DropdownContainer 
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <DropdownButton>
        {title}
        <KeyboardArrowDown />
      </DropdownButton>
      <DropdownContent isOpen={isOpen}>
        {items.map((item, index) => (
          <DropdownItem key={index} href={item.link}>
            {item.label}
          </DropdownItem>
        ))}
      </DropdownContent>
    </DropdownContainer>
  );
};

NavDropdown.propTypes = {
  title: PropTypes.string.isRequired,
  items: PropTypes.arrayOf(
    PropTypes.shape({
      link: PropTypes.string.isRequired,
      label: PropTypes.string.isRequired,
    })
  ).isRequired,
};