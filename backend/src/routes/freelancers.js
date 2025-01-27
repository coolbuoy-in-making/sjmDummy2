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

function parseSkills(skills) {
  if (!skills) return [];
  
  try {
    if (Array.isArray(skills)) {
      return skills.map(s => String(s).trim()).filter(Boolean);
    }
    
    if (typeof skills === 'string') {
      try {
        const parsed = JSON.parse(skills);
        if (Array.isArray(parsed)) {
          return parsed.map(s => String(s).trim()).filter(Boolean);
        }
      } catch {
        // If JSON parsing fails, try comma split
        return skills.split(',').map(s => s.trim()).filter(Boolean);
      }
    }
    
    return [];
  } catch (error) {
    console.error('Error parsing skills:', error);
    return [];
  }
}

router.get('/', async (req, res) => {
  try {
    console.log('Fetching freelancers...');
    
    const freelancers = await Freelancer.findAll({
      attributes: [
        'id', 'username', 'name', 'job_title', 'skills',
        'experience', 'rating', 'hourly_rate', 'profile_url',
        'availability', 'total_sales', 'desc'
      ],
      raw: true
    });

    const formattedFreelancers = freelancers.map(f => {
      try {
        // Use enhanced skill parsing
        const skills = parseSkills(f.skills);
        
        return {
          id: String(f.id),
          username: f.username || '',
          name: f.name || '',
          jobTitle: f.job_title || '',
          skills: skills, // Cleaned skills array
          experience: parseInt(f.experience || 0),
          rating: parseFloat(f.rating || 0),
          hourlyRate: parseFloat(f.hourly_rate || 0), 
          profileUrl: f.profile_url || `/profile/${f.id}`,
          availability: Boolean(f.availability),
          totalSales: parseInt(f.total_sales || 0),
          desc: f.desc || ''
        };
      } catch (error) {
        console.error(`Error formatting freelancer ${f.id}:`, error);
        return null;
      }
    }).filter(Boolean);

    console.log(`Returning ${formattedFreelancers.length} freelancers`);
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

router.post('/interview-request', auth, async (req, res) => {
  try {
    const { freelancerId, projectId, message } = req.body;
    
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    // Validate input
    if (!freelancerId || !projectId) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const freelancer = await Freelancer.findByPk(freelancerId);
    if (!freelancer) {
      return res.status(404).json({ error: 'Freelancer not found' });
    }

    const interview = await InterviewRequest.create({
      freelancerId,
      projectId,
      userId: req.user.id,
      message,
      status: 'pending',
      createdAt: new Date(),
      updatedAt: new Date()
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