import styled from 'styled-components';
import JobCard from '../shared/JobCard';

const Section = styled.section`
  padding: 80px 0;
  background: #f7faf7;
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 24px;
  margin-top: 40px;
`;

const mockJobs = [
  {
    id: 1,
    title: 'React Native Developer Needed',
    description: 'Looking for an experienced React Native developer for a long-term project...',
    budget: '5000',
    skills: ['React Native', 'JavaScript', 'Mobile Development']
  },
  // Add more jobs...
];

const FeaturedJobs = () => {
  return (
    <Section>
      <div className="container">
        <h2>Featured Jobs</h2>
        <Grid>
          {mockJobs.map(job => (
            <JobCard key={job.id} {...job} />
          ))}
        </Grid>
      </div>
    </Section>
  );
};

export default FeaturedJobs;
