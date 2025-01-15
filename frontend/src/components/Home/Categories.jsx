import styled from 'styled-components';
import { Link } from 'react-router-dom';
import developmentIcon from '../../../public/categories/development.svg';
import designIcon from '../../../public/categories/design.svg';
import writingIcon from '../../../public/categories/writing.svg';
import adminIcon from '../../../public/categories/admin.svg';
import marketingIcon from '../../../public/categories/marketing.svg';

const Section = styled.section`
  padding: 80px 0;
  background: white;
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 24px;
  margin-top: 40px;
`;

const CategoryCard = styled(Link)`
  padding: 24px;
  border: 1px solid #e4ebe4;
  border-radius: 16px;
  transition: all 0.2s;

  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
`;

const categories = [
  {
    title: "Development & IT",
    icon: developmentIcon,
    description: "Software engineers, developers & more"
  },
  {
    title: "Design & Creative",
    icon: designIcon,
    description: "Designers, animators & more"
  },
  {
    title: "Writing & Translation",
    icon: writingIcon,
    description: "Writers, translators & more"
  },
  {
    title: "Admin & Customer Support",
    icon: adminIcon,
    description: "Admins, virtual assistants & more"
  },
  {
    title: "Marketing",
    icon: marketingIcon,
    description: "Digital marketers, SEO & more"
  }
];

const Categories = () => {
  return (
    <Section>
      <div className="container">
        <h2>Browse talent by category</h2>
        <Grid>
          {categories.map(category => (
            <CategoryCard key={category.title} to="#">
              <img src={category.icon} alt={category.title} width={32} height={32} />
              <h3>{category.title}</h3>
              <p>{category.description}</p>
            </CategoryCard>
          ))}
        </Grid>
      </div>
    </Section>
  );
};

export default Categories;
