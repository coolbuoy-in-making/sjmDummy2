import { useState } from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';
import MenuIcon from '@mui/icons-material/Menu';
import SearchIcon from '@mui/icons-material/Search';
import CloseIcon from '@mui/icons-material/Close';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';

const Nav = styled.nav`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 64px;
  background: white;
  border-bottom: 1px solid ${props => props.theme.colors.lightGray};
  z-index: 1000;
`;

const NavContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 100%;
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 24px;
`;

const Logo = styled(Link)`
  height: 24px;
  margin-right: 32px;
  img {
    height: 100%;
  }
`;

const NavGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 24px;

  @media (max-width: ${props => props.theme.breakpoints.md}) {
    display: ${props => props.showOnMobile ? 'flex' : 'none'};
  }
`;

const NavLink = styled(Link)`
  color: ${props => props.theme.colors.dark};
  font-weight: 500;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 8px 12px;
  border-radius: 16px;
  transition: all 0.2s;

  &:hover {
    background: ${props => props.theme.colors.background};
  }
`;

const MobileMenu = styled.div`
  display: none;
  position: fixed;
  top: 64px;
  left: 0;
  right: 0;
  background: white;
  padding: 16px;
  border-top: 1px solid ${props => props.theme.colors.lightGray};
  box-shadow: ${props => props.theme.shadows.medium};

  @media (max-width: ${props => props.theme.breakpoints.md}) {
    display: ${props => props.isOpen ? 'block' : 'none'};
  }
`;

const MenuButton = styled.button`
  display: none;
  background: none;
  border: none;
  padding: 8px;
  
  @media (max-width: ${props => props.theme.breakpoints.md}) {
    display: block;
  }
`;

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Nav>
      <NavContainer>
        <MenuButton onClick={() => setIsOpen(!isOpen)}>
          {isOpen ? <CloseIcon /> : <MenuIcon />}
        </MenuButton>

        <NavGroup>
          <Logo to="/">
            <img src="/upwork-logo.svg" alt="Upwork" />
          </Logo>
          <NavLink to="/find-work">
            Find Work <KeyboardArrowDownIcon />
          </NavLink>
          <NavLink to="/find-talent">
            Find Talent <KeyboardArrowDownIcon />
          </NavLink>
          <NavLink to="/why-upwork">Why Upwork</NavLink>
          <NavLink to="/enterprise">Enterprise</NavLink>
        </NavGroup>

        <NavGroup showOnMobile>
          <SearchIcon style={{ cursor: 'pointer' }} />
          <NavLink to="/login">Log In</NavLink>
          <Link to="/signup" className="primary-button">
            Sign Up
          </Link>
        </NavGroup>

        <MobileMenu isOpen={isOpen}>
          <NavLink to="/find-work">Find Work</NavLink>
          <NavLink to="/find-talent">Find Talent</NavLink>
          <NavLink to="/why-upwork">Why Upwork</NavLink>
          <NavLink to="/enterprise">Enterprise</NavLink>
          <NavLink to="/login">Log In</NavLink>
          <Link to="/signup" className="primary-button" style={{ marginTop: '16px' }}>
            Sign Up
          </Link>
        </MobileMenu>
      </NavContainer>
    </Nav>
  );
};

export default Navbar;