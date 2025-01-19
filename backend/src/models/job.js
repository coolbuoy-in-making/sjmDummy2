module.exports = (sequelize, DataTypes) => {
  const Job = sequelize.define('Job', {
    title: {
      type: DataTypes.STRING,
      allowNull: false
    },
    desc: {
      type: DataTypes.TEXT,
      allowNull: false
    },
    budget: {
      type: DataTypes.DECIMAL(10, 2),
      allowNull: false
    },
    skills: {
      type: DataTypes.JSON,
      defaultValue: []
    },
    status: {
      type: DataTypes.ENUM('open', 'in_progress', 'completed'),
      defaultValue: 'open'
    },
    duration: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    complexity: {
      type: DataTypes.ENUM('low', 'medium', 'high'),
      defaultValue: 'medium'
    }
  });

  Job.associate = (models) => {
    Job.belongsTo(models.User, { foreignKey: 'clientId' });
    Job.hasMany(models.Proposal, { foreignKey: 'jobId' });
  };

  return Job;
};