import styled from 'styled-components';
import Card from '../shared/Card';

const FilterSection = styled.div`
  margin-bottom: 24px;
`;

const FilterTitle = styled.h3`
  font-size: 16px;
  margin-bottom: 16px;
`;

const CheckboxGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const Checkbox = styled.label`
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
`;

const JobFilters = ({ onFilterChange }) => {
  return (
    <Card>
      <FilterSection>
        <FilterTitle>Experience Level</FilterTitle>
        <CheckboxGroup>
          <Checkbox>
            <input type="checkbox" name="experience" value="entry" />
            Entry Level
          </Checkbox>
          <Checkbox>
            <input type="checkbox" name="experience" value="intermediate" />
            Intermediate
          </Checkbox>
          <Checkbox>
            <input type="checkbox" name="experience" value="expert" />
            Expert
          </Checkbox>
        </CheckboxGroup>
      </FilterSection>

      <FilterSection>
        <FilterTitle>Job Type</FilterTitle>
        <CheckboxGroup>
          <Checkbox>
            <input type="checkbox" name="jobType" value="hourly" />
            Hourly
          </Checkbox>
          <Checkbox>
            <input type="checkbox" name="jobType" value="fixed" />
            Fixed Price
          </Checkbox>
        </CheckboxGroup>
      </FilterSection>
    </Card>
  );
};

export default JobFilters;