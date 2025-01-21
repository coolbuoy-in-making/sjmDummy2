const { faker } = require('@faker-js/faker');
const bcrypt = require('bcryptjs');

const skillSets = {
  'Web Development': [
    'React', 'Angular', 'Vue.js', 'Node.js', 'Express.js', 'Django', 'Ruby on Rails',
    'PHP', 'Laravel', 'ASP.NET', 'Spring Boot', 'GraphQL', 'REST API', 'MongoDB',
    'PostgreSQL', 'MySQL', 'Redis', 'AWS', 'Docker', 'Kubernetes', 'CI/CD', 'Git',
    'TypeScript', 'Next.js', 'Gatsby', 'Svelte', 'WebSocket', 'Redux', 'MobX',
    'Webpack', 'Sass/SCSS', 'TailwindCSS', 'Bootstrap', 'Material UI', 'Three.js',
    'WebGL', 'Progressive Web Apps', 'Web Security', 'Performance Optimization'
  ],
  'Mobile Development': [
    'React Native', 'Flutter', 'iOS', 'Swift', 'Android', 'Kotlin', 'Java',
    'Xamarin', 'Unity', 'Mobile UI/UX', 'App Store Optimization', 'Push Notifications',
    'Mobile Security', 'Cross-Platform Development', 'SwiftUI', 'Objective-C',
    'Ionic', 'PhoneGap', 'Mobile Analytics', 'ARKit', 'CoreML', 'Firebase',
    'Mobile Game Development', 'Augmented Reality', 'Virtual Reality'
  ],
  'Data Science': [
    'Python', 'R', 'SQL', 'Machine Learning', 'Deep Learning', 'TensorFlow',
    'PyTorch', 'NLP', 'Computer Vision', 'Data Visualization', 'Tableau',
    'Power BI', 'Statistics', 'A/B Testing', 'Big Data', 'Hadoop', 'Spark',
    'SAS', 'SPSS', 'Data Mining', 'Predictive Analytics', 'Time Series Analysis',
    'Quantum Computing', 'Neural Networks', 'Reinforcement Learning', 'MLOps'
  ],
  'Digital Marketing': [
    'SEO', 'SEM', 'Content Marketing', 'Social Media Marketing',
    'Email Marketing', 'Google Ads', 'Facebook Ads', 'Analytics',
    'Marketing Automation', 'CRM', 'Lead Generation', 'Conversion Optimization',
    'TikTok Marketing', 'Instagram Marketing', 'LinkedIn Marketing',
    'Marketing Strategy', 'Brand Management', 'Influencer Marketing',
    'Video Marketing', 'Affiliate Marketing', 'Growth Hacking'
  ],
  'Data Analytics': [
    'Python', 'R', 'SQL', 'Machine Learning', 'Deep Learning', 'TensorFlow',
    'PyTorch', 'NLP', 'Computer Vision', 'Data Visualization', 'Tableau',
    'Power BI', 'Statistics', 'A/B Testing', 'Big Data', 'Hadoop', 'Spark',
    'SAS', 'SPSS', 'Data Mining', 'Predictive Analytics', 'Time Series Analysis',
    'Quantum Computing', 'Neural Networks', 'Reinforcement Learning', 'MLOps'
  ],
  'Graphics Design': [
    'UI/UX Design', 'Figma', 'Adobe XD', 'Sketch', 'InVision', 'Prototyping',
    'Wireframing', 'User Research', 'Design Systems', 'Visual Design',
    'Motion Design', 'Design Thinking', 'Accessibility', 'Photoshop',
    'Illustrator', 'After Effects', 'Premier Pro', 'Brand Identity',
    'Logo Design', '3D Modeling', 'Blender', 'Maya', 'ZBrush', 'Animation'
  ],
  'Digital Marketing': [
    'SEO', 'SEM', 'Content Marketing', 'Social Media Marketing',
    'Email Marketing', 'Google Ads', 'Facebook Ads', 'Analytics',
    'Marketing Automation', 'CRM', 'Lead Generation', 'Conversion Optimization',
    'TikTok Marketing', 'Instagram Marketing', 'LinkedIn Marketing',
    'Marketing Strategy', 'Brand Management', 'Influencer Marketing'
  ],
  'Writing & Translation': [
    'Content Writing', 'Copywriting', 'Technical Writing', 'Creative Writing',
    'Blog Writing', 'Article Writing', 'Research Writing', 'Academic Writing',
    'Grant Writing', 'Editing', 'Proofreading', 'Translation', 'Localization',
    'Transcription', 'Ghostwriting', 'SEO Writing', 'Medical Writing'
  ],
  'AI & Machine Learning': [
    'Deep Learning', 'Natural Language Processing', 'Computer Vision',
    'Reinforcement Learning', 'Neural Networks', 'Machine Learning Engineering',
    'AI Ethics', 'ChatGPT', 'LangChain', 'Prompt Engineering', 'GPT Integration',
    'AI Model Training', 'ML Ops', 'AI Application Development'
  ],
  'Blockchain & Cryptocurrency': [
    'Smart Contracts', 'Solidity', 'Ethereum', 'Web3.js', 'DeFi Development',
    'NFT Development', 'Cryptocurrency', 'Blockchain Architecture',
    'Smart Contract Auditing', 'Tokenomics', 'Crypto Trading Bots'
  ],
  'Business & Finance': [
    'Financial Analysis', 'Business Planning', 'Market Research',
    'Business Strategy', 'Investment Analysis', 'Risk Management',
    'Financial Modeling', 'Valuation', 'Business Intelligence',
    'Accounting', 'Bookkeeping', 'Tax Preparation', 'Business Consulting'
  ],
  'IT & Networking': [
    'Network Security', 'System Administration', 'Cloud Computing',
    'DevOps', 'AWS', 'Azure', 'Google Cloud', 'Linux Administration',
    'Cybersecurity', 'Virtualization', 'IT Support', 'Database Administration'
  ],
  'Engineering & Architecture': [
    'Mechanical Engineering', 'Electrical Engineering', 'Civil Engineering',
    'Chemical Engineering', 'AutoCAD', 'SolidWorks', 'MATLAB',
    'Circuit Design', 'PCB Design', 'IoT Development', '3D Printing'
  ],
  'Admin Support': [
    'Virtual Assistance', 'Data Entry', 'Customer Service',
    'Project Management', 'Office Administration', 'Transcription',
    'Calendar Management', 'Email Management', 'CRM Administration'
  ],
  'Legal Services': [
    'Contract Law', 'Intellectual Property', 'Corporate Law',
    'Legal Writing', 'Patent Law', 'Trademark Law', 'Legal Research',
    'Compliance', 'Legal Consultation', 'Document Review'
  ],
  'Video & Animation': [
    'Video Editing', 'Motion Graphics', '3D Animation',
    'Visual Effects', 'Video Production', 'Whiteboard Animation',
    'Character Animation', 'Explainer Videos', 'Commercial Production'
  ],
  'Music & Audio': [
    'Music Production', 'Sound Design', 'Voice Over',
    'Audio Editing', 'Podcast Production', 'Mixing & Mastering',
    'Composition', 'Sound Engineering', 'Audio Books'
  ],
  'Quality Assurance': [
    'Manual Testing', 'Automated Testing', 'Performance Testing',
    'Security Testing', 'Mobile Testing', 'API Testing',
    'Test Planning', 'Bug Tracking', 'Test Automation', 'QA Management'
  ],
  'Sales & Business Development': [
    'Sales Strategy', 'Lead Generation', 'Business Development',
    'Account Management', 'Sales Automation', 'CRM Management',
    'Sales Funnel Optimization', 'Partnership Development'
  ],
  'Education & Training': [
    'Online Teaching', 'Course Creation', 'Instructional Design',
    'Educational Content', 'Training Development', 'E-learning',
    'Curriculum Development', 'Learning Management Systems'
  ]
};

const generateProfileUrl = (id) => {
  return `${process.env.FRONTEND_URL || 'http://localhost:5173'}/profile/${id}`;
};
const generateUniqueId = (startId, i) => startId + i;

module.exports = {
  up: async (queryInterface, Sequelize) => {
    const password = await bcrypt.hash('password123', 10);
    const users = [];
    const freelancers = [];

    // Create clients first (IDs 1-400)
        for (let i = 0; i < 400; i++) {
          const userId = generateUniqueId(1, i);
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

    // Create 500 users
    for (let i = 0; i < 500; i++) {
      const userId = generateUniqueId(401, i);
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
        skills: JSON.stringify([
          ...faker.helpers.arrayElements(skillSets[category], 4),
          ...faker.helpers.arrayElements(
            skillSets[faker.helpers.arrayElement(Object.keys(skillSets))], 
            2
          )
        ]),
        hourlyRate: faker.number.int({ min: 15, max: 150 }),
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

    // Create 500 freelancer users and their freelancer profiles
    for (let i = 0; i < 500; i++) {
      const userId = generateUniqueId(501, i);
      const category = faker.helpers.arrayElement(Object.keys(skillSets));
      const FullName = faker.person.fullName();
      const username = faker.internet.username();
      const experience = faker.number.int({ min: 1, max: 15 });

      // Create corresponding freelancer profile
      freelancers.push({
        userId: userId,
        username,
        name: FullName,
        job_title: `${category} Specialist`,
        skills: [
          ...faker.helpers.arrayElements(skillSets[category], 4),
          ...faker.helpers.arrayElements(
            skillSets[faker.helpers.arrayElement(Object.keys(skillSets))], 
            2
          )
        ].join(','),
        experience,
        rating: faker.number.float({ min: 4.0, max: 5.0, precision: 0.1 }),
        hourly_rate: faker.number.int({ min: 25, max: 150 }),
        availability: faker.datatype.boolean(),
        profile_url: generateProfileUrl(userId),
        total_sales: faker.number.int({ min: 0, max: 100 }),
        desc: `${faker.person.jobDescriptor()} ${category} professional with ${experience} years of experience.`,
        createdAt: faker.date.past(),
        updatedAt: new Date()
      });
    }



    // Insert users first
    await queryInterface.bulkInsert('Users', users);

    // Get inserted users to map IDs
    const insertedUsers = await queryInterface.sequelize.query(
      `SELECT id FROM Users WHERE userType = 'freelancer' ORDER BY id ASC;`
    );
    const userIds = insertedUsers[0].map(u => u.id);

    // Add userId to freelancers
    freelancers.forEach((freelancer, index) => {
      freelancer.userId = userIds[index];
    });

    // Insert freelancer profiles
    return queryInterface.bulkInsert('Freelancers', freelancers);
  },
  down: (queryInterface, Sequelize) => {
    return queryInterface.bulkDelete('Users', null, {});
  }
};