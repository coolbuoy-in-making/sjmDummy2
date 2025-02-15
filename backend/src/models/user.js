'use strict';
const { Model } = require('sequelize');

module.exports = function(sequelize, DataTypes) {
  class User extends Model {
    static associate(models) {
      User.hasMany(models.Job, { foreignKey: 'clientId' });
      User.hasOne(models.Freelancer, {
        foreignKey: 'userId',
        as: 'freelancerProfile'
      });
    }
  }

  User.init({
    name: {
      type: DataTypes.STRING,
      allowNull: false
    },
    email: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: true
    },
    password: {
      type: DataTypes.STRING,
      allowNull: false
    },
    userType: {
      type: DataTypes.ENUM('client', 'freelancer'),
      allowNull: false
    },
    title: DataTypes.STRING,
    desc: DataTypes.TEXT,
    companySize: DataTypes.STRING,
    industry: DataTypes.STRING,
    skills: DataTypes.JSON,
    hourlyRate: DataTypes.DECIMAL(10, 2),
    profileUrl: DataTypes.STRING,
    totalJobs: {
      type: DataTypes.INTEGER,
      defaultValue: 0
    },
    successRate: {
      type: DataTypes.INTEGER,
      defaultValue: 0
    },
    totalEarnings: {
      type: DataTypes.DECIMAL(10, 2),
      defaultValue: 0
    },
    availability: {
      type: DataTypes.BOOLEAN,
      defaultValue: true
    },
    yearsOfExperience: DataTypes.INTEGER,
    education: DataTypes.JSON,
    certifications: DataTypes.JSON
  }, {
    sequelize,
    modelName: 'User'
  });

  return User;
};