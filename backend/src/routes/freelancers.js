const express = require('express');
const router = express.Router();
const { Freelancer, InterviewRequest } = require('../models');

// Middleware to verify API key
const verifyApiKey = (req, res, next) => {
  const apiKey = req.headers.authorization?.split(' ')[1];
  if (apiKey === process.env.API_KEY) {
    next();
  } else {
    res.status(401).json({ error: 'Invalid API key' });
  }
};

// Import auth middleware
const auth = require('../middleware/auth');

router.get('/', verifyApiKey, async (req, res) => {
  try {
    console.log('Received API key:', req.headers.authorization);

    const freelancers = await Freelancer.findAll({
      where: {
        availability: true  // Only get available freelancers
      },
      attributes: [
        'id', 'username', 'name', 'job_title', 'skills', 
        'experience', 'rating', 'hourly_rate', 'profile_url',
        'availability', 'total_sales', 'desc'
      ],
      raw: true
    });

    console.log('Raw freelancers data:', freelancers[0]); // Log first freelancer for debugging

    if (!freelancers || freelancers.length === 0) {
      console.log('No freelancers found, returning empty array');
      return res.json([]);
    }

    // Transform and validate the data
    const formattedFreelancers = freelancers.map(f => ({
      id: f.id,
      name: f.name,
      jobTitle: f.job_title,
      skills: f.skills ? f.skills.split(',').map(s => s.trim()) : [],
      hourlyRate: parseFloat(f.hourly_rate),
      rating: parseFloat(f.rating),
      experience: parseInt(f.experience),
      totalSales: parseInt(f.total_sales),
      availability: Boolean(f.availability),
      profile_url: `/profile/${f.id}`,
      desc: f.desc,
      matchDetails: {
        skillMatch: { 
          skills: [],
          count: 0
        },
        matchPercentage: 0
      }
    }));

    console.log('Sending formatted freelancer example:', formattedFreelancers[0]);
    res.json(formattedFreelancers);
  } catch (error) {
    console.error('Error fetching freelancers:', error);
    res.status(500).json({ error: error.message });
  }
});

// Update interview endpoint to use auth middleware instead of API key
router.post('/interview', auth, async (req, res) => {
  try {
    const { freelancerId, projectId, message } = req.body;
    
    // Add user validation
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const freelancer = await Freelancer.findByPk(freelancerId);
    if (!freelancer) {
      return res.status(404).json({ error: 'Freelancer not found' });
    }

    // Add user ID to interview request
    const interview = await InterviewRequest.create({
      freelancerId,
      projectId,
      userId: req.user.id,
      message,
      status: 'pending'
    });

    res.json({ 
      success: true, 
      interview: {
        id: interview.id,
        status: 'pending',
        message: 'Interview request sent successfully'
      }
    });
  } catch (error) {
    console.error('Error creating interview request:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;