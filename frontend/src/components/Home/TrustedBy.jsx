import styled from 'styled-components';

const Section = styled.section`
  padding: 40px 0;
  background: white;
`;

const Title = styled.p`
  color: #5e6d55;
  font-size: 14px;
  text-align: center;
  margin-bottom: 24px;
`;

const LogoGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 32px;
  align-items: center;
  justify-items: center;
`;

const CompanyLogo = styled.img`
  height: 32px;
  opacity: 0.7;
`;

const TrustedBy = () => {
  const companies = [
    'microsoft', 'airbnb', 'bissell', 'nasdaq', 'automatic'
  ];

  return (
    <Section>
      <div className="container">
        <Title>Trusted by</Title>
        <LogoGrid>
          {companies.map(company => (
            <CompanyLogo 
              key={company}
              src={`/companies/${company}.svg`} 
              alt={company}
            />
          ))}
        </LogoGrid>
      </div>
    </Section>
  );
};

export default TrustedBy;
