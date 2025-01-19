const { faker } = require('@faker-js/faker');
const bcrypt = require('bcryptjs');

const skillSets = {
  'Web Development': [
    'React', 'Angular', 'Vue.js', 'Node.js', 'Express.js', 'Django',
    'Ruby on Rails', 'PHP', 'Laravel', 'ASP.NET', 'Spring Boot',
    'GraphQL', 'REST API', 'MongoDB', 'PostgreSQL', 'MySQL',
    'Redis', 'AWS', 'Docker', 'Kubernetes', 'CI/CD', 'Git'
  ],
  'Mobile Development': [
    'React Native', 'Flutter', 'iOS', 'Swift', 'Android', 'Kotlin',
    'Java', 'Xamarin', 'Unity', 'Mobile UI/UX', 'App Store Optimization',
    'Push Notifications', 'Mobile Security', 'Cross-Platform Development'
  ],
  'Data Science': [
    'Python', 'R', 'SQL', 'Machine Learning', 'Deep Learning', 'TensorFlow',
    'PyTorch', 'NLP', 'Computer Vision', 'Data Visualization', 'Tableau',
    'Power BI', 'Statistics', 'A/B Testing', 'Big Data', 'Hadoop', 'Spark'
  ],
  'Design': [
    'UI/UX Design', 'Figma', 'Adobe XD', 'Sketch', 'InVision',
    'Prototyping', 'Wireframing', 'User Research', 'Design Systems',
    'Visual Design', 'Motion Design', 'Design Thinking', 'Accessibility'
  ],
  'Digital Marketing': [
    'SEO', 'SEM', 'Content Marketing', 'Social Media Marketing',
    'Email Marketing', 'Google Ads', 'Facebook Ads', 'Analytics',
    'Marketing Automation', 'CRM', 'Lead Generation', 'Conversion Optimization'
  ]
};

const generateProfileUrl = (name) => {
  return `${process.env.FRONTEND_URL || 'http://localhost:5173'}/users/${name.toLowerCase().replace(/\s+/g, '-')}`;
};

module.exports = {
  up: async (queryInterface, Sequelize) => {
    const password = await bcrypt.hash('password123', 10);
    const users = [];

    // Create 400 clients
    for (let i = 0; i < 400; i++) {
      const companySize = faker.helpers.arrayElement([
        '1-10', '11-50', '51-200', '201-500', '501-1000', '1000+'
      ]);

      users.push({
        name: faker.company.name(),
        email: faker.internet.email(),
        password,
        userType: 'client',
        title: `${faker.person.jobTitle()} at ${faker.company.name()}`,
        desc: `${faker.company.catchPhrase()} - ${faker.company.buzzPhrase()}`,
        companySize,
        industry: faker.helpers.arrayElement([
          'Software Development', 'Digital Marketing', 'E-commerce',
          'Healthcare', 'Education', 'Financial Services', 'Media',
          'Real Estate', 'Manufacturing', 'Technology', 'Retail'
        ]),
        totalJobs: faker.number.int({ min: 0, max: 50 }),
        totalEarnings: faker.number.float({ min: 0, max: 100000, precision: 2 }),
        createdAt: faker.date.past(),
        updatedAt: new Date()
      });
    }

    // Create 500 users
    for (let i = 0; i < 500; i++) {
      const category = faker.helpers.arrayElement(Object.keys(skillSets));
      const fullName = faker.person.fullName();
      const experience = faker.number.int({ min: 1, max: 15 });
      
      users.push({
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
        profileUrl: generateProfileUrl(fullName),
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

    return queryInterface.bulkInsert('Users', users);
  },
  down: (queryInterface, Sequelize) => {
    return queryInterface.bulkDelete('Users', null, {});
  }
};