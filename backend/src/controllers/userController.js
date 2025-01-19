const { User, Job, Proposal } = require('../models');

exports.getProfile = async (req, res) => {
  try {
    const user = await User.findByPk(req.params.id, {
      attributes: { 
        exclude: ['password'],
        include: [
          'name', 'title', 'desc', 'userType', 'skills',
          'hourlyRate', 'profileUrl', 'availability', 'totalSales'
        ]
      }
    });
    
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }
    
    res.json(user);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
}

exports.updateProfile = async (req, res) => {
  try {
    const user = await User.findByPk(req.user.id);
    await user.update(req.body);
    res.json(user);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};