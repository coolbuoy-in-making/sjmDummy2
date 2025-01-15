import styled from 'styled-components';
import Card from '../shared/Card';
const JobsGrid = styled.div`
  display: grid;
  gap: 16px;
`;

const JobCard = styled(Card)`
  padding: 24px;
  cursor: pointer;
  transition: transform 0.2s;

  &:hover {
    transform: translateY(-2px);
  }
`;

const JobHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 16px;
`;

const JobTitle = styled.h3`
  font-size: 18px;
  color: ${props => props.theme.colors.dark};
`;

const Budget = styled.span`
  font-weight: 500;
  color: ${props => props.theme.colors.dark};
`;

const JobDetails = styled.div`
  font-size: 14px;
  color: ${props => props.theme.colors.gray};
  margin-bottom: 16px;
`;

const Skills = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
`;

const Skill = styled.span`
  background: ${props => props.theme.colors.background};
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
`;

const mockJobs = [
  {
    id: 1,
    title: 'Senior React Developer Needed',
    hourlyRate: '$50-70',
    description: 'Looking for an experienced React developer for a long-term project...',
    skills: ['React', 'Redux', 'TypeScript'],
    postedDate: '2 hours ago'
  },
  // Add more mock jobs
];

const JobList = () => {
  return (
    <JobsGrid>
      {mockJobs.map(job => (
        <JobCard key={job.id} hoverable>
          <JobHeader>
            <JobTitle>{job.title}</JobTitle>
            <Budget>{job.hourlyRate}</Budget>
          </JobHeader>
          <JobDetails>
            <p>{job.description}</p>
            <p style={{ marginTop: '8px' }}>Posted {job.postedDate}</p>
          </JobDetails>
          <Skills>
            {job.skills.map(skill => (
              <Skill key={skill}>{skill}</Skill>
            ))}
          </Skills>
        </JobCard>
      ))}
    </JobsGrid>
  );
};

export default JobList;