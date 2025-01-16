require('dotenv').config();
const express = require('express');
const cors = require('cors');
const db = require('./src/models');
const authRoutes = require('./src/routes/auth');
const jobsRoutes = require('./src/routes/jobs');
const usersRoutes = require('./src/routes/users');

const app = express();

app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true
}));

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/jobs', jobsRoutes);
app.use('/api/users', usersRoutes);

const PORT = process.env.PORT || 5000;

db.sequelize.sync().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
});