module.exports = (sequelize, DataTypes) => {
    const User = sequelize.define('User', {
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
      skills: DataTypes.JSON,
      hourlyRate: DataTypes.DECIMAL(10, 2),
      description: DataTypes.TEXT
    });
  
    User.associate = (models) => {
      User.hasMany(models.Job, { foreignKey: 'clientId' });
      User.hasMany(models.Proposal, { foreignKey: 'freelancerId' });
    };
  
    return User;
  };