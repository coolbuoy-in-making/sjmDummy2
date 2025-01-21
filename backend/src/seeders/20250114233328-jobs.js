const { faker } = require('@faker-js/faker');

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

module.exports = {
  up: async (queryInterface, Sequelize) => {
    const clients = await queryInterface.sequelize.query(
      `SELECT id FROM Users WHERE userType = 'client';`
    );
    const clientIds = clients[0].map(client => client.id);

    const jobs = [];
    const categories = Object.keys(skillSets);

    // Create 1000 jobs
    for (let i = 0; i < 1000; i++) {
      const category = faker.helpers.arrayElement(categories);
      const duration = faker.number.int({ min: 1, max: 12 });
      
      jobs.push({
        title: `${category} - ${faker.company.catchPhrase()}`,
        desc: `
Project Overview:
${faker.lorem.paragraph()}

Key Requirements:
${faker.helpers.arrayElements(skillSets[category], 3).map(skill => `- ${skill}`).join('\n')}

Additional Details:
- Duration: ${duration} ${duration === 1 ? 'month' : 'months'}
- ${faker.lorem.sentences(2)}

Technical Requirements:
${faker.helpers.arrayElements(skillSets[category], 2).map(skill => `- Advanced knowledge of ${skill}`).join('\n')}

Nice to Have:
${faker.helpers.arrayElements(skillSets[faker.helpers.arrayElement(categories)], 2).map(skill => `- Experience with ${skill}`).join('\n')}
        `,
        budget: faker.number.int({ min: 500, max: 10000 }),
        skills: JSON.stringify(faker.helpers.arrayElements(skillSets[category], 4)),
        status: faker.helpers.arrayElement(['open', 'in_progress', 'completed']),
        clientId: faker.helpers.arrayElement(clientIds),
        duration: faker.number.int({ min: 1, max: 12 }),
        complexity: faker.helpers.arrayElement(['low', 'medium', 'high']),
        createdAt: faker.date.recent(90),
        updatedAt: new Date()
      });
    }

    return queryInterface.bulkInsert('Jobs', jobs);
  }
};