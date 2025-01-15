module.exports = (sequelize, DataTypes) => {
    const Proposal = sequelize.define('Proposal', {
      coverLetter: {
        type: DataTypes.TEXT,
        allowNull: false
      },
      bid: {
        type: DataTypes.DECIMAL(10, 2),
        allowNull: false
      },
      status: {
        type: DataTypes.ENUM('pending', 'accepted', 'rejected'),
        defaultValue: 'pending'
      },
      estimatedDuration: {
        type: DataTypes.INTEGER,
        allowNull: false
      }
    });
  
    Proposal.associate = (models) => {
      Proposal.belongsTo(models.User, { foreignKey: 'freelancerId' });
      Proposal.belongsTo(models.Job, { foreignKey: 'jobId' });
    };
  
    return Proposal;
  };