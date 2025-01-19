// backend/src/models/freelancer.js
module.exports = (sequelize, DataTypes) => {
  const Freelancer = sequelize.define('Freelancer', {
    username: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: true
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false
    },
    job_title: {
      type: DataTypes.STRING,
      allowNull: false
    },
    skills: {
      type: DataTypes.TEXT,
      allowNull: false,
      defaultValue: ''
    },
    experience: {
      type: DataTypes.INTEGER,
      defaultValue: 0
    },
    rating: {
      type: DataTypes.FLOAT,
      defaultValue: 0
    },
    hourly_rate: {
      type: DataTypes.FLOAT,
      defaultValue: 0
    },
    profile_url: {
      type: DataTypes.STRING
    },
    availability: {
      type: DataTypes.BOOLEAN,
      defaultValue: true
    },
    total_sales: {
      type: DataTypes.INTEGER,
      defaultValue: 0
    },
    desc: {
      type: DataTypes.TEXT,
      defaultValue: ''
    }
  });

  return Freelancer;
};