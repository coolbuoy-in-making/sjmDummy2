import styled from 'styled-components';
import SearchIcon from '@mui/icons-material/Search';
import TuneIcon from '@mui/icons-material/Tune';

const SearchWrapper = styled.div`
  display: flex;
  gap: 16px;
  margin-bottom: 32px;

  @media (max-width: ${props => props.theme.breakpoints.sm}) {
    flex-direction: column;
  }
`;

const SearchBar = styled.div`
  flex: 1;
  position: relative;
`;

const SearchInput = styled.input`
  width: 100%;
  padding: 12px 16px 12px 48px;
  border: 2px solid ${props => props.theme.colors.lightGray};
  border-radius: 24px;
  font-size: 16px;

  &:focus {
    border-color: ${props => props.theme.colors.primary};
    outline: none;
  }
`;

const IconWrapper = styled.div`
  position: absolute;
  left: 16px;
  top: 50%;
  transform: translateY(-50%);
  color: ${props => props.theme.colors.gray};
`;

const FilterButton = styled.button`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  border: 2px solid ${props => props.theme.colors.lightGray};
  border-radius: 24px;
  background: white;
  color: ${props => props.theme.colors.dark};
  font-weight: 500;

  &:hover {
    background: ${props => props.theme.colors.background};
  }
`;

const JobSearch = () => {
  return (
    <SearchWrapper>
      <SearchBar>
        <IconWrapper>
          <SearchIcon />
        </IconWrapper>
        <SearchInput placeholder="Search for jobs" />
      </SearchBar>
      <FilterButton>
        <TuneIcon /> Filters
      </FilterButton>
    </SearchWrapper>
  );
};

export default JobSearch;