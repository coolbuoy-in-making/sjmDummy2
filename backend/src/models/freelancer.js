module.exports = (sequelize, DataTypes) => {
  const Freelancer = sequelize.define('Freelancer', {
    userId: {
      type: DataTypes.INTEGER,
      references: {
        model: 'Users',
        key: 'id'
      }
    },
    username: DataTypes.STRING,
    name: DataTypes.STRING,
    job_title: DataTypes.STRING,
    skills: DataTypes.TEXT,
    experience: DataTypes.INTEGER,
    rating: DataTypes.FLOAT,
    hourly_rate: DataTypes.FLOAT,
    profile_url: DataTypes.STRING,
    availability: DataTypes.BOOLEAN,
    total_sales: DataTypes.INTEGER,
    desc: DataTypes.TEXT
  });

  Freelancer.associate = (models) => {
    Freelancer.belongsTo(models.User, {
      foreignKey: 'userId',
      as: 'user'
    });
  };

  return Freelancer;
};