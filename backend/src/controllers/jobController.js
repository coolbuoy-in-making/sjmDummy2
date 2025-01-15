const { Job, User, Proposal } = require('../models');

exports.createJob = async (req, res) => {
  try {
    const job = await Job.create({
      ...req.body,
      clientId: req.user.id
    });
    res.status(201).json(job);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};

exports.getAllJobs = async (req, res) => {
  try {
    const jobs = await Job.findAll({
      include: [{
        model: User,
        as: 'client',
        attributes: ['name', 'email']
      }]
    });
    res.json(jobs);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

exports.getJobById = async (req, res) => {
  try {
    const job = await Job.findByPk(req.params.id, {
      include: [{
        model: User,
        as: 'client',
        attributes: ['name', 'email']
      }]
    });
    if (!job) {
      return res.status(404).json({ message: 'Job not found' });
    }
    res.json(job);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};