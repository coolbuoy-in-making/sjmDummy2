const jwt = require('jsonwebtoken');
const { User } = require('../models');

const auth = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({ 
        error: 'No token provided',
        code: 'NO_TOKEN'
      });
    }

    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      const user = await User.findOne({ 
        where: { id: decoded.id },
        attributes: { exclude: ['password'] }
      });

      if (!user) {
        throw new Error('User not found');
      }

      req.token = token;
      req.user = user;
      next();
    } catch (error) {
      if (error.name === 'TokenExpiredError') {
        return res.status(401).json({ 
          error: 'Token expired',
          code: 'TOKEN_EXPIRED'
        });
      }
      throw error;
    }
  } catch (error) {
    console.error('Auth error:', error);
    res.status(401).json({ 
      message: 'Please authenticate',
      code: 'AUTH_FAILED'
    });
  }
};

module.exports = auth;