const { faker } = require('@faker-js/faker');

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