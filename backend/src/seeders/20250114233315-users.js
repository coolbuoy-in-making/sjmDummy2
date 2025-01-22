const { faker } = require('@faker-js/faker');
const bcrypt = require('bcryptjs');
const axios = require('axios');

// Enhanced skill sets with comprehensive categories
const skillSets = {
  'Engineering & Development': {
    'Software Development': [
      // Core Programming
      'JavaScript', 'Python', 'Java', 'C++', 'C#', 'Ruby', 'PHP', 'Swift', 'Go', 'Rust', 'Kotlin', 'TypeScript',
      // Web Frontend
      'React', 'Angular', 'Vue.js', 'Next.js', 'Svelte', 'HTML5', 'CSS3', 'SASS/SCSS', 'WebGL', 'Three.js',
      // Web Backend
      'Node.js', 'Django', 'Ruby on Rails', 'Laravel', 'ASP.NET Core', 'Spring Boot', 'Express.js', 'FastAPI',
      // Mobile
      'React Native', 'Flutter', 'iOS Development', 'Android Development', 'Xamarin', 'SwiftUI', 'Kotlin Multiplatform',
      // Database
      'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'DynamoDB', 'Cassandra', 'Oracle', 'SQL Server',
      // Cloud & DevOps
      'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins', 'GitLab CI', 'Terraform', 'Ansible'
    ],
    'Game Development': [
      'Unity', 'Unreal Engine', 'GameMaker Studio', 'Godot', 'C++ Game Dev', 'DirectX', 'OpenGL',
      'Game Design', 'Level Design', '3D Modeling for Games', 'Game AI', 'Game Physics', 'Multiplayer Networking'
    ],
    'Embedded Systems': [
      'Arduino', 'Raspberry Pi', 'Embedded C', 'RTOS', 'IoT Development', 'Microcontroller Programming',
      'FPGA', 'PCB Design', 'Electronics Design', 'Firmware Development'
    ]
  },

  'Data Science & AI': {
    'Machine Learning': [
      'TensorFlow', 'PyTorch', 'Scikit-learn', 'Deep Learning', 'Neural Networks', 'Computer Vision',
      'NLP', 'Reinforcement Learning', 'MLOps', 'Auto ML', 'Feature Engineering'
    ],
    'Data Engineering': [
      'Apache Spark', 'Hadoop', 'Airflow', 'Kafka', 'ETL', 'Data Warehousing', 'Data Modeling',
      'Big Data', 'Data Pipeline', 'Stream Processing'
    ],
    'Analytics': [
      'Data Analysis', 'Statistical Analysis', 'R Programming', 'Power BI', 'Tableau', 'Excel',
      'Google Analytics', 'A/B Testing', 'Business Intelligence', 'Predictive Analytics'
    ]
  },

  'Design & Creative': {
    'UI/UX Design': [
      'Figma', 'Adobe XD', 'Sketch', 'InVision', 'User Research', 'Wireframing', 'Prototyping',
      'Design Systems', 'Interaction Design', 'Mobile Design', 'Web Design', 'Responsive Design'
    ],
    'Graphic Design': [
      'Adobe Photoshop', 'Illustrator', 'InDesign', 'After Effects', 'Typography', 'Logo Design',
      'Brand Identity', 'Print Design', 'Packaging Design', 'Motion Graphics'
    ],
    '3D & Animation': [
      'Blender', 'Maya', 'Cinema 4D', '3D Modeling', 'Character Animation', 'Motion Capture',
      'VFX', 'Rigging', 'Texturing', 'Rendering'
    ]
  },

  'Business & Management': {
    'Project Management': [
      'Agile', 'Scrum', 'Kanban', 'PRINCE2', 'PMP', 'Program Management', 'Risk Management',
      'Stakeholder Management', 'Resource Planning', 'Project Planning', 'Jira', 'Trello'
    ],
    'Business Analysis': [
      'Requirements Analysis', 'Process Modeling', 'BPMN', 'UML', 'User Stories', 'Use Cases',
      'Gap Analysis', 'Business Process Improvement', 'Data Analysis', 'SQL'
    ],
    'Product Management': [
      'Product Strategy', 'Product Roadmap', 'User Stories', 'Market Research', 'Competitive Analysis',
      'Product Analytics', 'Growth Hacking', 'A/B Testing', 'User Feedback', 'Product Launch'
    ]
  },

  'Marketing & Content': {
    'Digital Marketing': [
      'SEO', 'SEM', 'Social Media Marketing', 'Content Marketing', 'Email Marketing', 'PPC',
      'Google Ads', 'Facebook Ads', 'Marketing Analytics', 'Marketing Automation'
    ],
    'Content Creation': [
      'Content Writing', 'Copywriting', 'Technical Writing', 'Blog Writing', 'Article Writing',
      'Editing', 'Proofreading', 'Content Strategy', 'SEO Writing'
    ],
    'Video & Audio': [
      'Video Editing', 'Video Production', 'Sound Design', 'Podcast Production', 'Voice Over',
      'Audio Editing', 'Motion Graphics', 'Color Grading', 'Screenwriting'
    ]
  },
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
    await initializeSkillSets();
    const password = await bcrypt.hash('password123', 10);
    const users = [];
    const freelancers = [];

    // Create clients (IDs 1-400)
    for (let i = 0; i < 400; i++) {
      const userId = 1 + i;
      const companyName = faker.company.name();
      
      users.push({
        id: userId,
        name: companyName,
        email: faker.internet.email(),
        password,
        userType: 'client',
        title: `${faker.person.jobTitle()} at ${companyName}`,
        desc: faker.company.catchPhrase(),
        companySize: faker.helpers.arrayElement([
          '1-10', '11-50', '51-200', '201-500', '501-1000', '1000+'
        ]),
        industry: faker.helpers.arrayElement([
          'Software Development', 'Digital Marketing', 'E-commerce',
          'Healthcare', 'Education', 'Financial Services'
        ]),
        totalJobs: faker.number.int({ min: 0, max: 50 }),
        totalEarnings: faker.number.float({ min: 0, max: 100000, precision: 2 }),
        createdAt: new Date(),
        updatedAt: new Date()
      });
    }

    // Create freelancer users (IDs 401-900)
    for (let i = 0; i < 500; i++) {
      const userId = 401 + i;
      const category = faker.helpers.arrayElement(Object.keys(skillSets));
      const fullName = faker.person.fullName();
      const experience = faker.number.int({ min: 1, max: 15 });
      
      users.push({
        id: userId,
        name: fullName,
        email: faker.internet.email(),
        password,
        userType: 'freelancer',
        title: `${category} Specialist`,
        skills: JSON.stringify(getRandomSkills(category)),
        hourlyRate: faker.number.int({ min: 25, max: 150 }),
        desc: `${faker.person.jobDescriptor()} ${category} professional with ${experience} years of experience. ${faker.lorem.paragraph()}`,
        profileUrl: generateProfileUrl(userId),
        totalJobs: faker.number.int({ min: 0, max: 100 }),
        successRate: faker.number.int({ min: 70, max: 100 }),
        totalEarnings: faker.number.float({ min: 1000, max: 200000, precision: 2 }),
        availability: faker.datatype.boolean(),
        yearsOfExperience: experience,
        education: JSON.stringify([{
          degree: faker.helpers.arrayElement([
            'Bachelor of Science', 'Master of Science', 'Bachelor of Arts'
          ]),
          field: faker.helpers.arrayElement([
            'Computer Science', 'Information Technology', 'Design',
            'Marketing', 'Business Administration'
          ]),
          school: faker.company.name() + ' University',
          year: 2024 - experience - faker.number.int({ min: 0, max: 5 })
        }]),
        certifications: JSON.stringify(
          faker.helpers.arrayElements([
            'AWS Certified Developer', 'Google Analytics', 'Scrum Master',
            'PMP', 'CISSP', 'CompTIA A+', 'Microsoft Azure'
          ], faker.number.int({ min: 1, max: 3 }))
        ),
        createdAt: faker.date.past(),
        updatedAt: new Date()
      });
    }

    // Create freelancer profiles
    for (let i = 0; i < 500; i++) {
      const userId = 401 + i;
      const category = faker.helpers.arrayElement(Object.keys(skillSets));
      const skills = getRandomSkills(category);
      
      freelancers.push({
        userId,
        username: faker.internet.userName(),
        name: users.find(u => u.id === userId).name,
        job_title: `${category} Specialist`,
        skills: skills.join(','),
        experience: faker.number.int({ min: 1, max: 15 }),
        rating: faker.number.float({ min: 4.0, max: 5.0, precision: 0.1 }),
        hourly_rate: faker.number.int({ min: 25, max: 150 }),
        availability: faker.datatype.boolean(),
        profile_url: generateProfileUrl(userId),
        total_sales: faker.number.int({ min: 0, max: 100 }),
        desc: `Expert ${category} professional. ${faker.lorem.paragraph()}`,
        createdAt: faker.date.past(),
        updatedAt: new Date()
      });
    }

    // Insert users and freelancers
    await queryInterface.bulkInsert('Users', users);
    return queryInterface.bulkInsert('Freelancers', freelancers);
  },

  down: async (queryInterface, Sequelize) => {
    await queryInterface.bulkDelete('Freelancers', null, {});
    return queryInterface.bulkDelete('Users', null, {});
  }
};