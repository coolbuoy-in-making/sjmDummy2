import styled from 'styled-components';

const PageContainer = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 24px;
  text-align: left;
`;

const Hero = styled.section`
  background: #13544e;
  color: white;
  padding: 80px 0;
  margin: -40px -24px 80px;
`;

const HeroContent = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 24px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
`;

const Section = styled.section`
  margin-bottom: 80px;
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 32px;
  margin-top: 40px;
`;

const Card = styled.div`
  background: white;
  padding: 32px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const ContactForm = styled.form`
  background: white;
  padding: 32px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const Enterprise = () => {
  return (
    <div>
      <Hero>
        <HeroContent>
          <div>
            <h1 style={{fontSize: '48px', marginBottom: '24px'}}>
              Enterprise Suite
            </h1>
            <p style={{fontSize: '20px', marginBottom: '32px'}}>
              Enterprise-grade solution to find, engage, and manage your workforce
            </p>
            <button className="primary-button">Contact us</button>
          </div>
          <img src="https://assets.upwork.com/enterprise-suite.webp" alt="Enterprise Suite" style={{width: '100%'}} />
        </HeroContent>
      </Hero>

      <PageContainer>
        <Section>
          <h2>Why Enterprise Suite</h2>
          <Grid>
            <Card>
              <h3>Expert matching</h3>
              <p>Get matched with pre-vetted talent and agencies by our specialized team.</p>
            </Card>
            <Card>
              <h3>Flexible workforce</h3>
              <p>Scale your team up or down with top talent and agencies on demand.</p>
            </Card>
            <Card>
              <h3>End-to-end solution</h3>
              <p>Manage your talent lifecycle from sourcing to payment in one place.</p>
            </Card>
          </Grid>
        </Section>

        <Section>
          <h2>Contact us</h2>
          <ContactForm>
            <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px'}}>
              <input type="text" placeholder="First name" style={{padding: '12px'}} />
              <input type="text" placeholder="Last name" style={{padding: '12px'}} />
              <input type="email" placeholder="Work email" style={{padding: '12px'}} />
              <input type="text" placeholder="Company" style={{padding: '12px'}} />
              <input type="tel" placeholder="Phone number" style={{padding: '12px'}} />
              <input type="text" placeholder="Job title" style={{padding: '12px'}} />
            </div>
            <button className="primary-button" style={{marginTop: '24px', width: '100%'}}>
              Contact us
            </button>
          </ContactForm>
        </Section>
      </PageContainer>
    </div>
  );
};

export default Enterprise;
