import styled from 'styled-components';
import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../utils/api';

const ProfilePage = styled.div`
  padding: 40px 24px;
  max-width: 1200px;
  margin: 0 auto;
`;

const ProfileHeader = styled.div`
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 24px;
  margin-bottom: 40px;
`;

const Profile = () => {
  const { id } = useParams();
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const response = await api.get(`/users/${id}`);
        setProfile(response.data);
      } catch (error) {
        console.error('Error fetching profile:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, [id]);

  if (loading) return <div>Loading...</div>;
  if (!profile) return <div>Profile not found</div>;

  return (
    <ProfilePage>
      <ProfileHeader>
        <img 
          src={`https://ui-avatars.com/api/?name=${profile.name}&size=128`}
          alt={profile.name}
          style={{ borderRadius: '50%' }}
        />
        <div>
          <h1>{profile.name}</h1>
          <h2>{profile.title}</h2>
          <p>{profile.desc}</p>
          {profile.userType === 'freelancer' && (
            <div>
              <p>${profile.hourlyRate}/hr</p>
              <p>{profile.totalSales} jobs completed</p>
              <p>{profile.availability ? 'Available for work' : 'Currently busy'}</p>
            </div>
          )}
        </div>
      </ProfileHeader>

      {profile.userType === 'freelancer' && (
        <div>
          <h3>Skills</h3>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {profile.skills?.map((skill, i) => (
              <span key={i} className="skill-tag">{skill}</span>
            ))}
          </div>
        </div>
      )}
    </ProfilePage>
  );
};

export default Profile;