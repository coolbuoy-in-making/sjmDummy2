const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');
const jwt = require('jsonwebtoken');
const { User } = require('../models');

router.post('/register', authController.register);
router.post('/login', authController.login);

router.post('/refresh', async (req, res) => {
  try {
    const { token } = req.body;
    
    if (!token) {
      return res.status(400).json({ error: 'Token is required' });
    }

    const decoded = jwt.decode(token);
    if (!decoded) {
      return res.status(400).json({ error: 'Invalid token' });
    }

    const user = await User.findOne({ 
      where: { id: decoded.id },
      attributes: { exclude: ['password'] }
    });

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    // Generate new token
    const newToken = jwt.sign(
      { id: user.id, email: user.email },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );

    res.json({ token: newToken });
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
});

module.exports = router;