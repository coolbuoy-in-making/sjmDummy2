const express = require('express');
const router = express.Router();
const { Freelancer } = require('../models');

// Middleware to verify API key
const verifyApiKey = (req, res, next) => {
  const apiKey = req.headers.authorization?.split(' ')[1];
  if (apiKey === process.env.API_KEY) {
    next();
  } else {
    res.status(401).json({ error: 'Invalid API key' });
  }
};

router.get('/', verifyApiKey, async (req, res) => {
  try {
    console.log('Received API key:', req.headers.authorization);

    const freelancers = await Freelancer.findAll({
      attributes: [
        'id', 'username', 'name', 'job_title', 'skills', 
        'experience', 'rating', 'hourly_rate', 'profile_url',
        'availability', 'total_sales', 'desc'
      ],
      raw: true
    });

    console.log('Raw freelancers:', freelancers);

    if (!freelancers || freelancers.length === 0) {
      console.log('No freelancers found, returning empty array');
      return res.json([]);
    }

    // Transform skills from string to array
    const formattedFreelancers = freelancers.map(f => ({
      ...f,
      skills: f.skills ? f.skills.split(',').map(s => s.trim()) : []
    }));

    console.log('Sending formatted freelancers:', formattedFreelancers);
    res.json(formattedFreelancers);
  } catch (error) {
    console.error('Error fetching freelancers:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;