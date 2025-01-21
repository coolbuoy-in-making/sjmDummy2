import styled from 'styled-components';
import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import api from '../utils/api';  // Update to use default import
import { FaStar, FaClock, FaCheckCircle } from 'react-icons/fa';
import Card from '../components/shared/Card';

const ProfilePage = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 24px;
`;

const ProfileGrid = styled.div`
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 32px;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const Sidebar = styled.div`
  position: sticky;
  top: 24px;
  height: fit-content;
`;

const Avatar = styled.div`
  width: 150px;
  height: 150px;
  border-radius: 50%;
  overflow: hidden;
  margin: 0 auto 24px;
  border: 3px solid ${props => props.theme.colors.primary};
  
  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
`;

const Stats = styled.div`
  display: grid;
  gap: 16px;
  margin: 24px 0;

  .stat-item {
    display: flex;
    align-items: center;
    gap: 8px;
    color: ${props => props.theme.colors.gray};
    
    .value {
      color: ${props => props.theme.colors.dark};
      font-weight: 500;
    }
  }
`;

const MainContent = styled.div`
  display: flex;
  flex-direction: column;
  gap: 32px;
`;

const ProfileHeader = styled.div`
  h1 {
    font-size: 32px;
    margin-bottom: 8px;
  }

  .title {
    font-size: 20px;
    color: ${props => props.theme.colors.primary};
    margin-bottom: 16px;
  }
`;

const RatingBadge = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 12px;
  background: ${props => props.theme.colors.background};
  border-radius: 16px;
  color: ${props => props.theme.colors.primary};
  font-weight: 500;
`;

const SkillsSection = styled.div`
  h3 {
    margin-bottom: 16px;
  }

  .skills-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
`;

const SkillTag = styled.span`
  padding: 6px 16px;
  background: ${props => props.theme.colors.background};
  border-radius: 20px;
  font-size: 14px;
  color: ${props => props.theme.colors.dark};
  transition: all 0.2s;

  &:hover {
    background: ${props => props.theme.colors.primary};
    color: white;
  }
`;

const AvailabilityBadge = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: ${props => props.available ? props.theme.colors.success + '20' : props.theme.colors.gray + '20'};
  color: ${props => props.available ? props.theme.colors.success : props.theme.colors.gray};
  border-radius: 20px;
  font-weight: 500;
  margin-top: 16px;
`;

const WorkHistory = styled.div`
  h3 {
    margin-bottom: 24px;
  }
`;

const JobCard = styled(Card)`
  margin-bottom: 16px;
  
  .job-title {
    font-size: 18px;
    font-weight: 500;
    margin-bottom: 8px;
  }

  .job-meta {
    display: flex;
    gap: 16px;
    color: ${props => props.theme.colors.gray};
    font-size: 14px;
    margin-bottom: 16px;
  }

  .job-description {
    color: ${props => props.theme.colors.dark};
    line-height: 1.6;
  }
`;

const Profile = () => {
  const { id } = useParams();
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        console.log('Fetching profile for ID:', id);
        // Update to use the correct API endpoint
        const response = await api.get(`/api/users/profile/${id}`);
        
        if (response.data) {
          console.log('Profile data received:', response.data);
          setProfile(response.data);
        }
      } catch (error) {
        console.error('Error fetching profile:', error.response || error);
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      fetchProfile();
    }
  }, [id]);

  if (loading) return <div>Loading...</div>;
  if (!profile) return <div>Profile not found</div>;

  return (
    <ProfilePage>
      <ProfileGrid>
        <Sidebar>
          <Avatar>
            <img 
              src={`https://ui-avatars.com/api/?name=${profile.name}&size=150`}
              alt={profile.name}
            />
          </Avatar>

          <Stats>
            <div className="stat-item">
              <FaStar />
              <span className="value">{profile.rating}% Job Success</span>
            </div>
            <div className="stat-item">
              <FaClock />
              <span className="value">${profile.hourlyRate}/hr</span>
            </div>
            <div className="stat-item">
              <FaCheckCircle />
              <span className="value">{profile.totalSales} Jobs Completed</span>
            </div>
          </Stats>

          <AvailabilityBadge available={profile.availability}>
            {profile.availability ? 'Available for work' : 'Currently busy'}
          </AvailabilityBadge>
        </Sidebar>

        <MainContent>
          <ProfileHeader>
            <h1>{profile.name}</h1>
            <div className="title">{profile.jobTitle}</div>
            <RatingBadge>
              <FaStar /> {profile.rating}% Success Rate
            </RatingBadge>
          </ProfileHeader>

          <div className="description">
            <h3>About</h3>
            <p>{profile.desc}</p>
          </div>

          <SkillsSection>
            <h3>Skills</h3>
            <div className="skills-grid">
              {profile.skills?.map((skill, i) => (
                <SkillTag key={i}>{skill}</SkillTag>
              ))}
            </div>
          </SkillsSection>

          <WorkHistory>
            <h3>Work History</h3>
            {profile.workHistory?.map((job, i) => (
              <JobCard key={i}>
                <div className="job-title">{job.title}</div>
                <div className="job-meta">
                  <span>{job.date}</span>
                  <span>${job.earnings}</span>
                  <span>{job.hours} hours</span>
                </div>
                <div className="job-description">{job.description}</div>
              </JobCard>
            ))}
          </WorkHistory>
        </MainContent>
      </ProfileGrid>
    </ProfilePage>
  );
};

export default Profile;