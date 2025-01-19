const { faker } = require('@faker-js/faker');

module.exports = {
  up: async (queryInterface, Sequelize) => {
    // Get client IDs
    const clients = await queryInterface.sequelize.query(
      `SELECT id FROM Users WHERE userType = 'client';`
    );
    const clientIds = clients[0].map(client => client.id);

    const jobs = [];

    // Create 50 jobs
    for (let i = 0; i < 50; i++) {
      jobs.push({
        title: faker.person.jobTitle(),
        desc: faker.lorem.paragraphs(3),
        budget: faker.number.int({ min: 500, max: 10000 }),
        skills: JSON.stringify([
          faker.person.jobType(),
          faker.person.jobType(),
          faker.person.jobType()
        ]),
        status: faker.helpers.arrayElement(['open', 'in_progress', 'completed']),
        clientId: clientIds[Math.floor(Math.random() * clientIds.length)],
        createdAt: new Date(),
        updatedAt: new Date()
      });
    }

    return queryInterface.bulkInsert('Jobs', jobs);
  },
  down: (queryInterface, Sequelize) => {
    return queryInterface.bulkDelete('Jobs', null, {});
  }
};