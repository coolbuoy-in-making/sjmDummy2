require('dotenv').config();
const express = require('express');
const path = require('path');
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

// Serve static files
app.use(express.static('public'));

// Update CORS configuration
app.use(cors({
  origin: [
      'http://localhost:5173',
      process.env.FRONTEND_URL,
      'http://localhost:8000',
      process.env.FLASK_URL
  ].filter(Boolean),  // This removes any null/undefined values
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


// Root route
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'backend.html'));
});

// Mount routes with /api prefix
app.use('/api/auth', authRoutes);
app.use('/api/users', usersRoutes);
app.use('/api/jobs', jobsRoutes);
app.use('/api/freelancers', freelancersRoutes);


// API 404 handler - for /api routes
app.use('/api/*', (req, res) => {
  res.status(404).json({ message: 'Not implemented' });
});

// Frontend 404 handler - for non-API routes
app.use('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'backend.html'));
});


// Add error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: err.message });
});


const PORT = process.env.PORT || 5000;
const isDevMode = process.env.NODE_ENV !== 'production';
const forceSync = process.env.FORCE_SYNC === 'true' || isDevMode;

// Add this function to handle database initialization
async function initializeDatabase() {
  try {
    if (forceSync) {
      // Disable foreign key checks
      await db.sequelize.query('SET FOREIGN_KEY_CHECKS = 0');

      // Drop all tables in correct order
      await db.sequelize.query('DROP TABLE IF EXISTS `InterviewRequests`');
      await db.sequelize.query('DROP TABLE IF EXISTS `Proposals`');
      await db.sequelize.query('DROP TABLE IF EXISTS `Jobs`');
      await db.sequelize.query('DROP TABLE IF EXISTS `Freelancers`');
      await db.sequelize.query('DROP TABLE IF EXISTS `Users`');

      // Enable foreign key checks
      await db.sequelize.query('SET FOREIGN_KEY_CHECKS = 1');

      // Sync database to create tables
      await db.sequelize.sync({ force: true });

      // Run seeders
      const { exec } = require('child_process');
      const maxBuffer = 1024 * 1024 * 10; // 10MB buffer

      exec('npx sequelize-cli db:seed:all', { maxBuffer }, (error, stdout, stderr) => {
        if (error) {
          console.error('Error seeding data:', error);
          return;
        }
        console.log('Database seeded successfully');
        if (stdout) console.log(stdout);
      });
    } else {
      // Just sync without force in non-dev mode
      await db.sequelize.sync();
    }

    // Start server after database is ready
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
      if (isDevMode) {
        console.log('Development mode active');
      }
    });

  } catch (error) {
    console.error('Database initialization error:', error);
    process.exit(1);
  }
}

// Initialize database and start server
initializeDatabase().catch(error => {
  console.error('Failed to start server:', error);
  process.exit(1);
});