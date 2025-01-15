const { User, Job, Proposal } = require('../models');

exports.getProfile = async (req, res) => {
  try {
    const user = await User.findByPk(req.user.id, {
      attributes: { exclude: ['password'] },
      include: [
        {
          model: Job,
          as: 'jobs',
          where: { clientId: req.user.id },
          required: false
        },
        {
          model: Proposal,
          as: 'proposals',
          where: { freelancerId: req.user.id },
          required: false
        }
      ]
    });
    res.json(user);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

exports.updateProfile = async (req, res) => {
  try {
    const user = await User.findByPk(req.user.id);
    await user.update(req.body);
    res.json(user);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};