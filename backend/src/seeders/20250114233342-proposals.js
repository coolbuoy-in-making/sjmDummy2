const { faker } = require('@faker-js/faker');

module.exports = {
  up: async (queryInterface, Sequelize) => {
    const users = await queryInterface.sequelize.query(
      `SELECT id FROM Users WHERE userType = 'freelancer';`
    );
    const jobs = await queryInterface.sequelize.query(
      `SELECT id, budget FROM Jobs WHERE status = 'open';`
    );

    const freelancerIds = users[0].map(f => f.id);
    const jobsData = jobs[0];

    const proposals = [];

    for (let i = 0; i < 100; i++) {
      const job = faker.helpers.arrayElement(jobsData);
      const jobBudget = parseFloat(job.budget);
      
      proposals.push({
        coverLetter: `
Dear Hiring Manager,

${faker.lorem.paragraph()}

Why I'm the right fit:
${faker.lorem.sentences(3)}

My relevant experience:
- ${faker.lorem.sentence()}
- ${faker.lorem.sentence()}
- ${faker.lorem.sentence()}

Timeline and Approach:
${faker.lorem.paragraph()}

Looking forward to discussing this opportunity further.

Best regards,
        `,
        bid: faker.number.int({ 
          min: jobBudget * 0.8, 
          max: jobBudget * 1.2 
        }),
        status: faker.helpers.arrayElement(['pending', 'accepted', 'rejected']),
        estimatedDuration: faker.number.int({ min: 7, max: 60 }),
        freelancerId: faker.helpers.arrayElement(freelancerIds),
        jobId: job.id,
        createdAt: faker.date.recent(14),
        updatedAt: new Date()
      });
    }

    return queryInterface.bulkInsert('Proposals', proposals);
  }
};