require('dotenv').config();
const express = require('express');
const cors = require('cors');
const db = require('./src/models');
const authRoutes = require('./src/routes/auth');
const jobsRoutes = require('./src/routes/jobs');
const usersRoutes = require('./src/routes/users');
const freelancersRoutes = require('./src/routes/freelancers');

const app = express();

// Add middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Update CORS configuration
app.use(cors({
  origin: [
    'http://localhost:5173',  // Frontend
    'http://localhost:8000'   // AI service
  ],
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

// Add request logging
app.use((req, res, next) => {
  console.log(`${req.method} ${req.url}`, {
    headers: req.headers,
    body: req.body
  });
  next();
});

// Add before your routes
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path}`, {
    params: req.params,
    query: req.query,
    body: req.body,
    path: req.path,
    originalUrl: req.originalUrl
  });
  next();
});

// Mount routes with /api prefix
app.use('/api/auth', authRoutes);
app.use('/api/users', usersRoutes);
app.use('/api/jobs', jobsRoutes);
app.use('/api/freelancers', freelancersRoutes);

// Add catch-all route for debugging
app.use((req, res) => {
  console.log('404 - Route not found:', {
    method: req.method,
    url: req.url,
    params: req.params,
    query: req.query
  });
  res.status(404).json({ message: 'Route not found' });
});

// Add error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: err.message });
});

const PORT = process.env.PORT || 5000;
const isDevMode = process.env.NODE_ENV !== 'production';
const forceSync = process.env.FORCE_SYNC === 'true' || isDevMode;

db.sequelize.sync({ force: forceSync }).then(async () => {
  if (forceSync) {
      try {
        await db.sequelize.query('SET FOREIGN_KEY_CHECKS = 0');
        await db.sequelize.query('TRUNCATE TABLE Users');
        await db.sequelize.query('TRUNCATE TABLE Jobs');
        await db.sequelize.query('TRUNCATE TABLE Proposals');
        await db.sequelize.query('SET FOREIGN_KEY_CHECKS = 1');
        
        // Run seeders
        const { exec } = require('child_process');
        const maxBuffer = 1024 * 1024 * 10; // 10MB buffer
        
        const seedCommand = 'npx sequelize-cli db:seed:all';
        exec(seedCommand, { maxBuffer }, (error, stdout, stderr) => {
          if (error) {
            console.error('Error seeding data:', error);
            return;
          }
          console.log('Database seeded successfully');
          console.log(stdout);
        });
      } catch (error) {
        console.error('Error in sync/seed process:', error);
      }
  }

  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    // if (isDevMode) {
    //   console.log('Development mode: Database tables recreated');
    // }
  });
});