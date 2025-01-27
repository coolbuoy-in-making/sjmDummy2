'use strict';
const { Model } = require('sequelize');

module.exports = function(sequelize, DataTypes) {
  class Freelancer extends Model {
    static associate(models) {
      Freelancer.belongsTo(models.User, {
        foreignKey: 'userId',
        as: 'user'
      });
    }
  }

  Freelancer.init({
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
    skills: {
      type: DataTypes.TEXT,
      get() {
        const val = this.getDataValue('skills');
        if (!val) return [];
        try {
          // Handle array case
          if (Array.isArray(val)) return val;
          
          // Handle JSON string
          const parsed = JSON.parse(val);
          return Array.isArray(parsed) ? parsed : [];
          
        } catch (e) {
          // Handle plain string case
          if (typeof val === 'string') {
            return val.split(',').map(s => s.trim()).filter(Boolean);
          }
          console.error('Error getting skills:', e);
          return [];
        }
      },
      set(val) {
        try {
          if (!val) {
            this.setDataValue('skills', '[]');
            return;
          }
          
          let skillsArray;
          if (Array.isArray(val)) {
            skillsArray = val;
          } else if (typeof val === 'string') {
            try {
              skillsArray = JSON.parse(val);
            } catch {
              skillsArray = val.split(',').map(s => s.trim()).filter(Boolean);
            }
          } else {
            skillsArray = [];
          }
          
          // Ensure array and clean values
          skillsArray = Array.isArray(skillsArray) ? skillsArray : [];
          skillsArray = skillsArray.map(s => String(s).trim()).filter(Boolean);
          
          this.setDataValue('skills', JSON.stringify(skillsArray));
        } catch (e) {
          console.error('Error setting skills:', e);
          this.setDataValue('skills', '[]');
        }
      }
    },
    experience: DataTypes.INTEGER,
    rating: DataTypes.FLOAT,
    hourly_rate: DataTypes.FLOAT,
    profile_url: DataTypes.STRING,
    availability: DataTypes.BOOLEAN,
    total_sales: DataTypes.INTEGER,
    desc: DataTypes.TEXT
  }, {
    sequelize,
    modelName: 'Freelancer'
  });

  return Freelancer;
};