// backend/src/models/freelancer.js
module.exports = (sequelize, DataTypes) => {
    const Freelancer = sequelize.define('Freelancer', {
      id: {
        type: DataTypes.UUID,
        defaultValue: DataTypes.UUIDV4,
        primaryKey: true
      },
      name: DataTypes.STRING,
      job_title: DataTypes.STRING,
      skills: DataTypes.TEXT,
      experience: DataTypes.INTEGER,
      rating: DataTypes.FLOAT,
      hourly_rate: DataTypes.FLOAT,
      profile_url: DataTypes.STRING,
      availability: DataTypes.BOOLEAN,
      total_sales: DataTypes.INTEGER
    });
    return Freelancer;
  };