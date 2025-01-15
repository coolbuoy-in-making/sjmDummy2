const { faker } = require('@faker-js/faker');
const bcrypt = require('bcryptjs');

module.exports = {
  up: async (queryInterface, Sequelize) => {
    const password = await bcrypt.hash('password123', 10);
    const users = [];

    // Create 10 clients
    for (let i = 0; i < 10; i++) {
      users.push({
        name: faker.person.fullName(),
        email: faker.internet.email(),
        password,
        userType: 'client',
        title: faker.person.jobTitle(),
        description: faker.company.catchPhrase(),
        createdAt: new Date(),
        updatedAt: new Date()
      });
    }

    // Create 20 freelancers
    for (let i = 0; i < 20; i++) {
      users.push({
        name: faker.person.fullName(),
        email: faker.internet.email(),
        password,
        userType: 'freelancer',
        title: faker.person.jobTitle(),
        skills: JSON.stringify([
          faker.person.jobType(),
          faker.person.jobType(),
          faker.person.jobType()
        ]),
        hourlyRate: faker.number.int({ min: 15, max: 150 }),
        description: faker.lorem.paragraph(),
        createdAt: new Date(),
        updatedAt: new Date()
      });
    }

    return queryInterface.bulkInsert('Users', users);
  },
  down: (queryInterface, Sequelize) => {
    return queryInterface.bulkDelete('Users', null, {});
  }
};