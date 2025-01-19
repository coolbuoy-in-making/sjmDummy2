'use strict';

module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.createTable('Jobs', {
      id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      title: {
        type: Sequelize.STRING,
        allowNull: false
      },
      desc: {
        type: Sequelize.TEXT,
        allowNull: false
      },
      budget: {
        type: Sequelize.DECIMAL(10, 2),
        allowNull: false
      },
      skills: {
        type: Sequelize.JSON,
        defaultValue: []
      },
      status: {
        type: Sequelize.ENUM('open', 'in_progress', 'completed'),
        defaultValue: 'open'
      },
      duration: {
        type: Sequelize.INTEGER,
        allowNull: false
      },
      complexity: {
        type: Sequelize.ENUM('low', 'medium', 'high'),
        defaultValue: 'medium'
      },
      clientId: {
        type: Sequelize.INTEGER,
        references: {
          model: 'Users',
          key: 'id'
        },
        onUpdate: 'CASCADE',
        onDelete: 'CASCADE'
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
  }
};