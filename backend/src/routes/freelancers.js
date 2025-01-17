const express = require('express');
const router = express.Router();
const { Freelancer } = require('../models');
const auth = require('../middleware/auth');

router.get('/', auth, async (req, res) => {
  try {
    const freelancers = await Freelancer.findAll({
      attributes: [
        'id', 'name', 'job_title', 'skills', 
        'experience', 'rating', 'hourly_rate',
        'profile_url', 'availability', 'total_sales'
      ]
    });
    res.json(freelancers);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;