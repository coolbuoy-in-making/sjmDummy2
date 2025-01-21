const express = require('express');
const router = express.Router();
const { User, Freelancer } = require('../models');  // Add models import

// Debug middleware
router.use('/profile/:id', (req, res, next) => {
  console.log('Profile request:', {
    id: req.params.id,
    path: req.path,
    originalUrl: req.originalUrl
  });
  next();
});

// Update route pattern to ensure ID is captured
router.get('/profile/:id', async (req, res) => {
  try {
    const { id } = req.params;
    console.log('Looking up profile for ID:', id);

    // First check Freelancer model
    const freelancer = await Freelancer.findOne({
      where: { id },
      attributes: [
        'id', 'username', 'name', 'job_title', 'skills',
        'experience', 'rating', 'hourly_rate', 'profile_url',
        'availability', 'total_sales', 'desc'
      ],
      raw: true
    });

    if (freelancer) {
      console.log('Found freelancer:', freelancer);
      return res.json({
        ...freelancer,
        userType: 'freelancer',
        skills: freelancer.skills ? freelancer.skills.split(',').map(s => s.trim()) : [],
        hourlyRate: parseFloat(freelancer.hourly_rate)
      });
    }

    // If not found in Freelancer, check User model
    const user = await User.findByPk(id, {
      attributes: { exclude: ['password'] },
      raw: true
    });

    if (!user) {
      console.log('No profile found for ID:', id);
      return res.status(404).json({ message: 'Profile not found' });
    }

    console.log('Found user:', user);
    res.json(user);
  } catch (error) {
    console.error('Profile lookup error:', error);
    res.status(500).json({ message: error.message });
  }
});

module.exports = router;