import styled from 'styled-components';
import Card from './Card';

const JobTitle = styled.h3`
  font-size: 16px;
  color: #001e00;
  margin-bottom: 8px;
`;

const JobDetails = styled.div`
  font-size: 14px;
  color: #5e6d55;
  margin-bottom: 16px;
`;

const Budget = styled.span`
  font-weight: 500;
  color: #001e00;
`;

const Skills = styled.div`
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
`;

const Skill = styled.span`
  background: #f2f7f2;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
`;

const JobCard = ({ title, description, budget, skills }) => {
  return (
    <Card>
      <JobTitle>{title}</JobTitle>
      <JobDetails>
        <p>{description}</p>
        <p>Budget: <Budget>${budget}</Budget></p>
      </JobDetails>
      <Skills>
        {skills.map(skill => (
          <Skill key={skill}>{skill}</Skill>
        ))}
      </Skills>
    </Card>
  );
};

export default JobCard;
