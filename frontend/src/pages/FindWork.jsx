import styled from 'styled-components';
import JobSearch from '../components/Jobs/JobSearch';
import JobFilters from '../components/Jobs/JobFilters';
import JobList from '../components/Jobs/JobList';

const PageContainer = styled.div`
  padding: 40px 0;
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 24px;
  margin-top: 32px;
`;

const FindWork = () => {
  return (
    <PageContainer>
      <JobSearch />
      <Grid>
        <JobFilters />
        <JobList />
      </Grid>
    </PageContainer>
  );
};

export default FindWork;