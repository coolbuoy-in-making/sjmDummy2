const express = require('express');
const router = express.Router();
const userController = require('../controllers/userController');
const auth = require('../middleware/auth');

router.get('/profile/:id', auth, userController.getProfile);
router.put('/profile', auth, userController.updateProfile);

module.exports = router;