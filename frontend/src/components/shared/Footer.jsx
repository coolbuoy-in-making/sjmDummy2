import styled from 'styled-components';
import { Link } from 'react-router-dom';

const FooterContainer = styled.footer`
  background: #001e00;
  color: white;
  padding: 64px 0;
`;

const FooterGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 40px;
`;

const FooterColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

const FooterTitle = styled.h4`
  font-size: 16px;
  margin-bottom: 8px;
`;

const FooterLink = styled(Link)`
  color: #91e6b3;
  font-size: 14px;
  &:hover {
    text-decoration: underline;
  }
`;

const Footer = () => {
  return (
    <FooterContainer>
      <div className="container">
        <FooterGrid>
          <FooterColumn>
            <FooterTitle>For Clients</FooterTitle>
            <FooterLink to="/hire">How to Hire</FooterLink>
            <FooterLink to="/talent">Talent Marketplace</FooterLink>
            <FooterLink to="/projects">Project Catalog</FooterLink>
          </FooterColumn>
          <FooterColumn>
            <FooterTitle>For Talent</FooterTitle>
            <FooterLink to="/work">Find Work</FooterLink>
            <FooterLink to="/direct-contracts">Direct Contracts</FooterLink>
            <FooterLink to="/help">Help Center</FooterLink>
          </FooterColumn>
          <FooterColumn>
            <FooterTitle>Resources</FooterTitle>
            <FooterLink to="/help-support">Help & Support</FooterLink>
            <FooterLink to="/success">Success Stories</FooterLink>
            <FooterLink to="/reviews">Reviews</FooterLink>
          </FooterColumn>
        </FooterGrid>
      </div>
    </FooterContainer>
  );
};

export default Footer;
