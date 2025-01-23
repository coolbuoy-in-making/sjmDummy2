'use strict';

const { faker } = require('@faker-js/faker');
const bcrypt = require('bcryptjs');
const axios = require('axios');

// Comprehensive job titles organized by industry
const jobTitles = {
  'Technology & Software': [
    // Software Development
    'Senior Software Engineer', 'Full Stack Developer', 'Frontend Developer', 'Backend Developer',
    'Mobile App Developer', 'DevOps Engineer', 'Site Reliability Engineer', 'Cloud Solutions Architect',
    'Systems Architect', 'API Developer', 'Software Development Manager', 'Technical Lead',
    'QA Engineer', 'Test Automation Engineer', 'Security Engineer', 'Database Administrator',
    'Machine Learning Engineer', 'Blockchain Developer', 'iOS Developer', 'Android Developer',
    'Enterprise Architect', 'Solutions Engineer', 'Platform Engineer', 'Release Engineer',

    // Data Science & Analytics
    'Data Scientist', 'Data Engineer', 'Business Intelligence Analyst', 'Data Architect',
    'Analytics Engineer', 'Big Data Engineer', 'MLOps Engineer', 'AI Research Scientist',
    'Quantitative Analyst', 'Data Analytics Manager', 'Statistical Programmer',
    'NLP Engineer', 'Computer Vision Engineer', 'Deep Learning Specialist',

    // Infrastructure & Security
    'Network Engineer', 'Systems Administrator', 'Information Security Analyst',
    'Cloud Infrastructure Engineer', 'Network Security Engineer', 'IT Support Specialist',
    'Infrastructure Manager', 'Security Operations Engineer', 'Penetration Tester',
    'Cloud Security Engineer', 'Identity Access Manager', 'IT Project Manager'
  ],

  'Design & Creative': [
    // UX/UI Design
    'UX Designer', 'UI Designer', 'Product Designer', 'Senior Product Designer',
    'Design Systems Engineer', 'Interaction Designer', 'Visual Designer', 'UX Researcher',
    'Design Manager', 'Creative Director', 'Art Director', 'Design Lead',
    'Service Designer', 'Experience Designer', 'Digital Product Designer',

    // Graphics & Multimedia
    'Graphic Designer', 'Motion Designer', 'Brand Designer', '3D Artist',
    'Character Artist', 'Environment Artist', 'Technical Artist', 'VFX Artist',
    'Animation Director', 'Creative Technologist', 'Multimedia Designer',
    'Package Designer', 'Production Artist', 'Digital Illustrator',

    // Game Design
    'Game Designer', 'Level Designer', 'Game Artist', 'Character Designer',
    'Environment Designer', 'Narrative Designer', 'Game UI Designer', 'Concept Artist',
    'Technical Game Designer', 'Game Economy Designer', 'Game Systems Designer'
  ],

  'Marketing & Communications': [
    // Digital Marketing
    'Digital Marketing Manager', 'SEO Specialist', 'Content Marketing Manager',
    'Social Media Manager', 'Growth Marketing Manager', 'Email Marketing Specialist',
    'Marketing Analytics Manager', 'Paid Search Manager', 'Performance Marketing Manager',
    'Brand Marketing Manager', 'Product Marketing Manager', 'Campaign Manager',

    // Content & Communications
    'Content Strategist', 'Technical Writer', 'Copywriter', 'Content Designer',
    'Communications Manager', 'PR Manager', 'Brand Strategist', 'Editorial Director',
    'Content Operations Manager', 'Documentation Manager', 'Knowledge Manager',

    // Market Research
    'Market Research Analyst', 'Consumer Insights Manager', 'Market Intelligence Analyst',
    'Competitive Intelligence Analyst', 'Marketing Data Analyst', 'Survey Researcher'
  ],

  'Business & Management': [
    // Product Management
    'Product Manager', 'Senior Product Manager', 'Technical Product Manager',
    'Product Owner', 'Program Manager', 'Project Manager', 'Scrum Master',
    'Agile Coach', 'Product Operations Manager', 'Chief Product Officer',

    // Business Operations
    'Business Analyst', 'Management Consultant', 'Operations Manager',
    'Process Improvement Manager', 'Strategy Analyst', 'Business Development Manager',
    'Sales Operations Manager', 'Revenue Operations Manager', 'Customer Success Manager',

    // Finance & Strategy
    'Financial Analyst', 'Business Intelligence Manager', 'Strategic Planning Manager',
    'Operations Analyst', 'Risk Manager', 'Portfolio Manager', 'Investment Analyst'
  ],

  'Sales & Customer Service': [
    // Sales
    'Account Executive', 'Sales Manager', 'Business Development Representative',
    'Sales Development Representative', 'Enterprise Account Manager', 'Solutions Consultant',
    'Sales Engineer', 'Channel Partner Manager', 'Regional Sales Director',

    // Customer Service
    'Customer Success Manager', 'Account Manager', 'Customer Support Engineer',
    'Technical Support Specialist', 'Client Services Manager', 'Customer Experience Manager',
    'Implementation Specialist', 'Customer Onboarding Specialist'
  ],

  'Healthcare & Sciences': [
    // Healthcare Technology
    'Healthcare Systems Analyst', 'Clinical Data Manager', 'Health Informatics Specialist',
    'Medical Software Developer', 'Healthcare Project Manager', 'Clinical Systems Analyst',
    
    // Research & Development
    'Research Scientist', 'Clinical Research Coordinator', 'Biostatistician',
    'Laboratory Manager', 'Research Engineer', 'Clinical Data Analyst',
    'Regulatory Affairs Specialist', 'Quality Assurance Manager'
  ],
};

// Enhanced comprehensive skill sets
const skillSets = {
  'Software Development': {
    'Programming Languages': [
      'JavaScript', 'Python', 'Java', 'C++', 'C#', 'Ruby', 'PHP', 'Swift', 'Go', 'Rust',
      'Kotlin', 'TypeScript', 'Scala', 'R', 'MATLAB', 'Perl', 'Haskell', 'Dart',
      'Objective-C', 'Shell Scripting', 'PowerShell', 'VBA', 'Groovy', 'Lua'
    ],
    'Web Technologies': [
      'HTML5', 'CSS3', 'SASS/SCSS', 'WebAssembly', 'WebGL', 'WebRTC', 'PWAs',
      'Service Workers', 'Web Components', 'Web Security', 'OAuth', 'JWT',
      'RESTful APIs', 'GraphQL', 'WebSockets', 'Server-Sent Events'
    ],
    'Frontend Development': [
      'React', 'Angular', 'Vue.js', 'Next.js', 'Nuxt.js', 'Svelte', 'Redux',
      'MobX', 'Zustand', 'Tailwind CSS', 'Material-UI', 'Bootstrap',
      'Webpack', 'Vite', 'Babel', 'ESLint', 'Jest', 'Testing Library',
      'Storybook', 'Cypress', 'Playwright'
    ],
    'Backend Development': [
      'Node.js', 'Django', 'Ruby on Rails', 'Spring Boot', 'ASP.NET Core',
      'Laravel', 'Express.js', 'FastAPI', 'NestJS', 'Flask', 'Phoenix',
      'Gin', 'gRPC', 'Microservices', 'Message Queues', 'Caching',
      'API Gateway', 'Service Mesh'
    ],
    'Database & Storage': [
      'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Elasticsearch',
      'Cassandra', 'DynamoDB', 'Neo4j', 'InfluxDB', 'TimescaleDB',
      'SQL', 'NoSQL', 'GraphQL', 'Data Modeling', 'Database Design',
      'Database Optimization', 'Database Administration'
    ],
    'DevOps & Cloud': [
      'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Terraform',
      'Ansible', 'Jenkins', 'GitLab CI', 'GitHub Actions', 'CircleCI',
      'Prometheus', 'Grafana', 'ELK Stack', 'Datadog', 'New Relic',
      'Cloud Architecture', 'Infrastructure as Code', 'Site Reliability Engineering'
    ]
  },

  'Data Science & AI': {
    'Machine Learning': [
      'TensorFlow', 'PyTorch', 'Scikit-learn', 'XGBoost', 'LightGBM',
      'Neural Networks', 'Deep Learning', 'Computer Vision', 'NLP',
      'Reinforcement Learning', 'Time Series Analysis', 'Anomaly Detection',
      'Feature Engineering', 'Model Deployment', 'MLOps', 'AutoML'
    ],
    'Data Engineering': [
      'Apache Spark', 'Hadoop', 'Airflow', 'Kafka', 'Snowflake', 'dbt',
      'ETL/ELT', 'Data Warehousing', 'Data Lakes', 'Data Modeling',
      'Data Pipeline', 'Stream Processing', 'Batch Processing',
      'Data Quality', 'Data Governance', 'Data Catalog'
    ],
    'Analytics & Visualization': [
      'Tableau', 'Power BI', 'Looker', 'D3.js', 'Python Visualization',
      'R Visualization', 'Dashboard Design', 'Statistical Analysis',
      'A/B Testing', 'Experimentation', 'Business Intelligence',
      'Data Storytelling', 'Metric Definition'
    ]
  },

  'Design & Creative': {
    'Design Tools': [
      'Figma', 'Adobe XD', 'Sketch', 'InVision', 'Photoshop',
      'Illustrator', 'After Effects', 'Premiere Pro', 'Blender',
      'Maya', 'Cinema 4D', 'Unity', 'Unreal Engine', 'ZBrush'
    ],
    'Design Skills': [
      'UI Design', 'UX Design', 'Interaction Design', 'Visual Design',
      'Typography', 'Color Theory', 'Layout Design', 'Icon Design',
      'Design Systems', 'Wireframing', 'Prototyping', 'User Research',
      'Usability Testing', 'Information Architecture', 'Service Design'
    ],
    'Motion & 3D': [
      'Motion Graphics', 'Character Animation', '3D Modeling',
      'Texturing', 'Rigging', 'VFX', 'Compositing',
      'Particle Systems', 'Lighting', 'Rendering'
    ]
  },

  'Business & Product': {
    'Product Management': [
      'Product Strategy', 'Product Roadmap', 'User Stories',
      'Product Requirements', 'Market Research', 'Competitive Analysis',
      'Product Analytics', 'Growth Strategy', 'Product Marketing',
      'Launch Planning', 'Pricing Strategy', 'Product Operations'
    ],
    'Project Management': [
      'Agile', 'Scrum', 'Kanban', 'Waterfall', 'PRINCE2', 'PMP',
      'Risk Management', 'Stakeholder Management', 'Resource Planning',
      'Budget Management', 'Change Management', 'Program Management'
    ],
    'Business Analysis': [
      'Requirements Analysis', 'Process Modeling', 'BPMN', 'UML',
      'Business Process Improvement', 'Gap Analysis', 'Cost-Benefit Analysis',
      'Process Documentation', 'Quality Assurance', 'Compliance'
    ]
  },

  'Marketing & Content': {
    'Digital Marketing': [
      'SEO', 'SEM', 'Google Ads', 'Facebook Ads', 'LinkedIn Ads',
      'Content Marketing', 'Email Marketing', 'Marketing Automation',
      'Growth Marketing', 'Conversion Optimization', 'Attribution Modeling',
      'Marketing Analytics', 'Customer Segmentation'
    ],
    'Content Creation': [
      'Content Strategy', 'Copywriting', 'Technical Writing',
      'Blog Writing', 'White Papers', 'Case Studies',
      'Social Media Content', 'Video Scripts', 'Email Campaigns',
      'Content Operations', 'Editorial Planning'
    ],
    'Analytics & Tools': [
      'Google Analytics', 'Google Tag Manager', 'Adobe Analytics',
      'Mixpanel', 'Amplitude', 'HubSpot', 'Marketo', 'Salesforce',
      'SEMrush', 'Ahrefs', 'Mailchimp', 'Customer.io'
    ]
  }
};


// Function to enrich skill sets using GitHub Topics API
async function enrichSkillSetsFromGitHub() {
  try {
    const response = await axios.get('https://api.github.com/search/topics', {
      params: { q: 'topic:programming language framework library' },
      headers: { Accept: 'application/vnd.github.mercy-preview+json' }
    });

    const topics = response.data.items.map(item => item.name);
    skillSets['Engineering & Development']['Software Development'].push(...topics);
  } catch (error) {
    console.error('Failed to fetch GitHub topics:', error);
  }
}

// Function to enrich skill sets using Stack Exchange Tags API
async function enrichSkillSetsFromStackOverflow() {
  try {
    const response = await axios.get(
      'https://api.stackexchange.com/2.3/tags?order=desc&sort=popular&site=stackoverflow'
    );

    const tags = response.data.items.map(item => item.name);
    skillSets['Engineering & Development']['Software Development'].push(...tags);
  } catch (error) {
    console.error('Failed to fetch Stack Overflow tags:', error);
  }
}

// Initialize skill sets with external data
async function initializeSkillSets() {
  await Promise.all([
    enrichSkillSetsFromGitHub(),
    enrichSkillSetsFromStackOverflow()
  ]);

  // Remove duplicates and normalize skills
  Object.keys(skillSets).forEach(category => {
    Object.keys(skillSets[category]).forEach(subcategory => {
      skillSets[category][subcategory] = [...new Set(
        skillSets[category][subcategory].map(skill => skill.trim())
      )];
    });
  });
}

// Helper function to get all skills for a category
const getAllSkillsForCategory = (category) => {
  const skills = [];
  Object.values(skillSets[category]).forEach(subcategorySkills => {
    skills.push(...subcategorySkills);
  });
  return skills;
};

// Helper function to get profile URL
const generateProfileUrl = (id) => {
  return `${process.env.FRONTEND_URL || 'http://localhost:5173'}/profile/${id}`;
};

// Helper function to combine skills
const getRandomSkills = (category, count = 6) => {
  const primarySkills = faker.helpers.arrayElements(getAllSkillsForCategory(category), Math.min(4, count));
  const otherCategory = faker.helpers.arrayElement(Object.keys(skillSets));
  const secondarySkills = faker.helpers.arrayElements(getAllSkillsForCategory(otherCategory), Math.min(2, count - primarySkills.length));
  return [...new Set([...primarySkills, ...secondarySkills])];
};

module.exports = {
  up: async (queryInterface, Sequelize) => {
    try {
      const password = await bcrypt.hash('password123', 10);
      const users = [];
      const freelancers = [];

      // Get all available categories from our defined skillSets
      const availableCategories = Object.keys(skillSets);
      const availableIndustries = Object.keys(jobTitles);

      // Create clients (IDs 1-400)
      for (let i = 0; i < 400; i++) {
        const userId = i + 1;
        const industry = faker.helpers.arrayElement(availableIndustries);
        const jobTitlesForIndustry = jobTitles[industry];
        const companyName = faker.company.name();
        
        users.push({
          id: userId,
          name: companyName,
          email: faker.internet.email(),
          password,
          userType: 'client',
          title: faker.helpers.arrayElement(jobTitlesForIndustry), // Use defined job titles
          companySize: faker.helpers.arrayElement([
            '1-10', '11-50', '51-200', '201-500', '501-1000', '1000+'
          ]),
          industry,
          desc: faker.company.catchPhrase(),
          skills: JSON.stringify([]),
          hourlyRate: null,
          profileUrl: null,
          totalJobs: faker.number.int({ min: 0, max: 50 }),
          successRate: null,
          totalEarnings: faker.number.float({ min: 0, max: 100000, precision: 2 }),
          availability: true,
          yearsOfExperience: null,
          education: null,
          certifications: null,
          createdAt: new Date(),
          updatedAt: new Date()
        });
      }

      // Create freelancer users (IDs 401-900)
      for (let i = 0; i < 500; i++) {
        const userId = 401 + i;
        // Select a random category from our defined skillSets
        const category = faker.helpers.arrayElement(availableCategories);
        
        // Get all skills for this category
        const categorySkills = getAllSkillsForCategory(category);
        
        // Select random skills from the category
        const selectedSkills = faker.helpers.arrayElements(
          categorySkills,
          faker.number.int({ min: 3, max: 6 })
        );

        // Get matching job titles for the category
        const matchingIndustry = availableIndustries.find(ind => 
          ind.toLowerCase().includes(category.toLowerCase()) ||
          category.toLowerCase().includes(ind.toLowerCase())
        ) || availableIndustries[0];

        const fullName = faker.person.fullName();
        const experience = faker.number.int({ min: 1, max: 15 });
        const username = faker.internet.userName();
        const hourlyRate = faker.number.int({ min: 25, max: 150 });

        const user = {
          id: userId,
          name: fullName,
          email: faker.internet.email(),
          password,
          userType: 'freelancer',
          // Use job titles that match the skills category
          title: faker.helpers.arrayElement(jobTitles[matchingIndustry]),
          desc: `Professional ${category} specialist with ${experience} years of experience. ${faker.lorem.paragraph()}`,
          skills: JSON.stringify(selectedSkills),
          hourlyRate,
          profileUrl: `/profile/${userId}`,
          totalJobs: faker.number.int({ min: 0, max: 100 }),
          successRate: faker.number.int({ min: 70, max: 100 }),
          totalEarnings: faker.number.float({ min: 1000, max: 200000, precision: 2 }),
          availability: faker.datatype.boolean(),
          yearsOfExperience: experience,
          education: JSON.stringify([{
            degree: faker.helpers.arrayElement([
              'Bachelor of Science', 'Master of Science', 'Bachelor of Arts'
            ]),
            field: category,
            school: `${faker.company.name()} University`,
            year: 2024 - experience - faker.number.int({ min: 0, max: 5 })
          }]),
          certifications: JSON.stringify([
            'AWS Certified Developer',
            'Professional Certification',
            'Industry Certification'
          ]),
          createdAt: faker.date.past(),
          updatedAt: new Date()
        };

        const freelancer = {
          userId,
          username,
          name: fullName,
          job_title: user.title,
          skills: JSON.stringify(selectedSkills),
          experience,
          rating: faker.number.float({ min: 4.0, max: 5.0, precision: 0.1 }),
          hourly_rate: hourlyRate,
          availability: user.availability,
          profile_url: user.profileUrl,
          total_sales: user.totalJobs,
          desc: user.desc,
          createdAt: user.createdAt,
          updatedAt: user.updatedAt
        };

        users.push(user);
        freelancers.push(freelancer);
      }

      // Insert data with transaction
      const transaction = await queryInterface.sequelize.transaction();
      try {
        await queryInterface.bulkInsert('Users', users, { transaction });
        await queryInterface.bulkInsert('Freelancers', freelancers, { transaction });
        await transaction.commit();
        console.log('Successfully seeded Users and Freelancers');
      } catch (err) {
        await transaction.rollback();
        throw err;
      }

      return Promise.resolve();
    } catch (error) {
      console.error('Error in seeding:', error);
      return Promise.reject(error);
    }
  },

  down: async (queryInterface) => {
    const transaction = await queryInterface.sequelize.transaction();
    try {
      await queryInterface.bulkDelete('Freelancers', null, { transaction });
      await queryInterface.bulkDelete('Users', null, { transaction });
      await transaction.commit();
      return Promise.resolve();
    } catch (error) {
      await transaction.rollback();
      console.error('Error in reverting seed:', error);
      return Promise.reject(error);
    }
  }
};