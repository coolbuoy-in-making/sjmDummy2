const { faker } = require('@faker-js/faker');

module.exports = {
  up: async (queryInterface, Sequelize) => {
    // Get freelancer and job IDs
    const users = await queryInterface.sequelize.query(
      `SELECT id FROM Users WHERE userType = 'freelancer';`
    );
    const jobs = await queryInterface.sequelize.query(
      `SELECT id FROM Jobs WHERE status = 'open';`
    );

    const freelancerIds = users[0].map(f => f.id);
    const jobIds = jobs[0].map(j => j.id);

    const proposals = [];

    // Create 100 proposals
    for (let i = 0; i < 100; i++) {
      proposals.push({
        coverLetter: faker.lorem.paragraphs(2),
        bid: faker.number.int({ min: 100, max: 5000 }),
        status: faker.helpers.arrayElement(['pending', 'accepted', 'rejected']),
        estimatedDuration: faker.number.int({ min: 5, max: 90 }),
        freelancerId: freelancerIds[Math.floor(Math.random() * freelancerIds.length)],
        jobId: jobIds[Math.floor(Math.random() * jobIds.length)],
        createdAt: new Date(),
        updatedAt: new Date()
      });
    }

    return queryInterface.bulkInsert('Proposals', proposals);
  },
  down: (queryInterface, Sequelize) => {
    return queryInterface.bulkDelete('Proposals', null, {});
  }
};