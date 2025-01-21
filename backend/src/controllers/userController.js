const { User, Freelancer } = require('../models');

exports.getProfile = async (req, res) => {
  try {
    const { id } = req.params;
    console.log('Fetching profile for ID:', id);

    // First try to find in Freelancers table
    const freelancer = await Freelancer.findOne({
      where: { id },
      attributes: { 
        exclude: ['password'],
        include: [
          'name', 'job_title', 'skills', 'experience',
          'rating', 'hourly_rate', 'profile_url', 
          'availability', 'total_sales', 'desc'
        ]
      },
      raw: true
    });

    if (freelancer) {
      console.log('Found freelancer:', freelancer.name);
      return res.json({
        ...freelancer,
        userType: 'freelancer',
        skills: freelancer.skills.split(',').map(s => s.trim())
      });
    }

    // If not found in Freelancers, try Users table
    const user = await User.findByPk(id, {
      attributes: { exclude: ['password'] },
      raw: true
    });

    if (!user) {
      console.log('No profile found for ID:', id);
      return res.status(404).json({ message: 'Profile not found' });
    }

    console.log('Found user:', user.name);
    res.json(user);

  } catch (error) {
    console.error('Error in getProfile:', error);
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