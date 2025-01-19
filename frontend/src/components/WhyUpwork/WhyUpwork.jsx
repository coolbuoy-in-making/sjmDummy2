import styled from 'styled-components';

const PageContainer = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 24px;
  text-align: left;
`;

const Hero = styled.section`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
  margin-bottom: 80px;
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
  padding: 24px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const WhyUpwork = () => {
  return (
    <PageContainer>
      <Hero>
        <div>
          <h1>Why businesses turn to Upwork</h1>
          <p style={{fontSize: '20px', marginTop: '24px'}}>
            Discover why leading businesses choose Upwork for their most important projects.
          </p>
        </div>
        <img src="https://assets.upwork.com/why-upwork.webp" alt="Why Upwork" style={{width: '100%'}} />
      </Hero>

      <Section>
        <h2>Proof of quality</h2>
        <Grid>
          <Card>
            <h3>Trusted by leading brands</h3>
            <p>We work with 30% of Fortune 100 companies and top brands.</p>
          </Card>
          <Card>
            <h3>Verified work history</h3>
            <p>Check reviews, work history, and portfolio samples before hiring.</p>
          </Card>
          <Card>
            <h3>Protected payments</h3>
            <p>Payment protection and dispute assistance for clients and users.</p>
          </Card>
        </Grid>
      </Section>

      <Section>
        <h2>Success stories</h2>
        <Grid>
          <Card>
            <img src="https://assets.upwork.com/microsoft-logo.svg" alt="Microsoft" style={{height: '40px'}} />
            <p>"Upwork enables us to differentiate ourselves from our competitors and produce content at a higher caliber."</p>
            <small>Microsoft</small>
          </Card>
          {/* Add more success stories */}
        </Grid>
      </Section>

      <Section>
        <h2>Browse talent by category</h2>
        <Grid>
          {[
            'Development & IT',
            'Design & Creative',
            'Sales & Marketing',
            'Writing & Translation',
            'Admin & Customer Support',
            'Finance & Accounting'
          ].map(category => (
            <Card key={category}>
              <h3>{category}</h3>
              <button className="secondary-button">Browse {category}</button>
            </Card>
          ))}
        </Grid>
      </Section>
    </PageContainer>
  );
};

export default WhyUpwork;
