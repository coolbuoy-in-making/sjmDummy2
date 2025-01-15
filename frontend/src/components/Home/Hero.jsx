import styled from 'styled-components';
import { Link } from 'react-router-dom';

const HeroSection = styled.section`
  min-height: calc(100vh - 64px);
  padding: 80px 0;
  background: ${props => props.theme.colors.background};
  overflow: hidden;

  @media (max-width: ${props => props.theme.breakpoints.md}) {
    padding: 40px 0;
    text-align: center;
  }
`;

const HeroContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  gap: 48px;
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 24px;

  @media (max-width: ${props => props.theme.breakpoints.md}) {
    grid-template-columns: 1fr;
    gap: 32px;
  }
`;

const HeroContent = styled.div`
  max-width: 600px;
  margin: 0 auto;

  @media (max-width: ${props => props.theme.breakpoints.md}) {
    max-width: 100%;
  }
`;

const HeroImage = styled.img`
  width: 100%;
  max-width: 600px;
  height: auto;
  object-fit: contain;

  @media (max-width: ${props => props.theme.breakpoints.md}) {
    max-width: 400px;
    margin: 0 auto;
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 16px;
  margin-top: 32px;

  @media (max-width: ${props => props.theme.breakpoints.sm}) {
    flex-direction: column;
  }
`;

const Hero = () => {
  return (
    <HeroSection>
      <HeroContainer>
        <HeroContent>
          <h1>How work should work</h1>
          <p style={{ fontSize: '1.25rem', marginTop: '24px', color: props => props.theme.colors.gray }}>
            Forget the old rules. You can have the best people. Right now. Right here.
          </p>
          <ButtonGroup>
            <Link to="/find-talent" className="primary-button">
              Find Talent
            </Link>
            <Link to="/find-work" className="secondary-button">
              Find Work
            </Link>
          </ButtonGroup>
        </HeroContent>
        <HeroImage 
          src="/hero-image.webp" 
          alt="Upwork makes work easier"
          loading="eager"
        />
      </HeroContainer>
    </HeroSection>
  );
};

export default Hero;