import styled from 'styled-components';

const Section = styled.section`
  padding: 80px 0;
  background: white;
`;

const Steps = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 40px;
  margin-top: 40px;
`;

const Step = styled.div`
  text-align: center;
  padding: 24px;
`;

const StepNumber = styled.div`
  width: 48px;
  height: 48px;
  border-radius: 24px;
  background: #14a800;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  margin: 0 auto 16px;
`;

const HowItWorks = () => {
  return (
    <Section>
      <div className="container">
        <h2>How it works</h2>
        <Steps>
          <Step>
            <StepNumber>1</StepNumber>
            <h3>Post a job</h3>
            <p>Tell us what you need. Provide as many details as possible.</p>
          </Step>
          <Step>
            <StepNumber>2</StepNumber>
            <h3>Choose talent</h3>
            <p>Review proposals, interview favorites, and select the best.</p>
          </Step>
          <Step>
            <StepNumber>3</StepNumber>
            <h3>Start working</h3>
            <p>Begin the project and get results you love.</p>
          </Step>
        </Steps>
      </div>
    </Section>
  );
};

export default HowItWorks;
