'use strict';

module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.createTable('Users', {
      id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      name: {
        type: Sequelize.STRING,
        allowNull: false
      },
      email: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: true
      },
      password: {
        type: Sequelize.STRING,
        allowNull: false
      },
      userType: {
        type: Sequelize.ENUM('client', 'freelancer'),
        allowNull: false
      },
      title: {
        type: Sequelize.STRING
      },
      desc: {
        type: Sequelize.TEXT
      },
      companySize: {
        type: Sequelize.STRING
      },
      industry: {
        type: Sequelize.STRING
      },
      skills: {
        type: Sequelize.JSON
      },
      hourlyRate: {
        type: Sequelize.DECIMAL(10, 2)
      },
      profileUrl: {
        type: Sequelize.STRING
      },
      totalJobs: {
        type: Sequelize.INTEGER,
        defaultValue: 0
      },
      successRate: {
        type: Sequelize.INTEGER,
        defaultValue: 0
      },
      totalEarnings: {
        type: Sequelize.DECIMAL(10, 2),
        defaultValue: 0
      },
      availability: {
        type: Sequelize.BOOLEAN,
        defaultValue: true
      },
      yearsOfExperience: {
        type: Sequelize.INTEGER
      },
      education: {
        type: Sequelize.JSON
      },
      certifications: {
        type: Sequelize.JSON
      },
      createdAt: {
        allowNull: false,
        type: Sequelize.DATE
      },
      updatedAt: {
        allowNull: false,
        type: Sequelize.DATE
      }
    });
  },
  down: async (queryInterface, Sequelize) => {
    await queryInterface.dropTable('Users');
  }
};