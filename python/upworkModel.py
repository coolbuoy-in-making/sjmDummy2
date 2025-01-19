import os
from dotenv import load_dotenv
import sys
import uuid
import json
import logging
from typing import List, Dict, Optional, Any, Union
import numpy as np
import nltk
nltk.download('punkt_tab')
import requests
from datetime import datetime
import asyncio
import aiohttp

from sjm import (
    SkillsExtract, 
    user, 
    Project, 
    Server, 
    normalize_csv, 
    MatchingEngine,
    CollaborativeModel
)

# Add missing dict method to user class in sjm.py
def dict(self) -> Dict[str, Any]:
    return {
        'id': self.id,
        'username': self.username,
        'name': self.name,
        'job_title': self.job_title,
        'skills': self.skills,
        'experience': self.experience,
        'rating': self.rating,
        'hourly_rate': self.hourly_rate,
        'profile_url': self.profile_url,
        'availability': self.availability,
        'total_sales': self.total_sales
    }

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('upwork_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from config import Config

class UpworkIntegrationModel:
    def __init__(self, api_url: str, api_key: str):
        """Initialize with API configuration"""
        self.api_url = api_url
        self.api_key = api_key
        self.skill_extractor = SkillsExtract()
        self.users = None
        self.matching_engine = None
        self.custom_weights = {
            'content': 0.3,
            'collaborative': 0.3,
            'experience': 0.2,
            'rating': 0.1,
            'hourly_rate': 0.1,
            'top_rated': 0.0
        }

    async def load_users(self) -> List[user]:
        """Load users from backend API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                
                url = f'{self.api_url}/api/users'
                logger.info(f"Fetching users from {url}")
                logger.debug(f"Using headers: {headers}")
                
                try:
                    async with session.get(
                        url,
                        headers=headers,
                        timeout=30
                    ) as response:
                        response_text = await response.text()
                        logger.debug(f"Response status: {response.status}")
                        logger.debug(f"Response body: {response_text}")

                        # Debug API key
                        logger.debug(f"Using API key: {self.api_key}")

                        if response.status == 401:
                            logger.error(f"Authentication failed. API Key: {self.api_key}")
                            if Config.DEBUG:
                                return self._get_mock_users()
                            raise Exception("Authentication failed. Please check your API key.")

                        if response.status != 200:
                            logger.error(f"API returned unexpected status {response.status}")
                            if Config.DEBUG:
                                return self._get_mock_users()
                            raise Exception(f"API Error: Status {response.status}")

                        try:
                            data = json.loads(response_text)
                            logger.info(f"Successfully loaded {len(data)} users")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON response: {e}")
                            if Config.DEBUG:
                                return self._get_mock_users()
                            raise

                        users = []
                        for item in data:
                            try:
                                user = user(
                                    id=str(item.get('id', '')),
                                    username=str(item.get('username', '')),
                                    name=str(item.get('name', '')),
                                    job_title=str(item.get('job_title', '')),
                                    skills=item.get('skills', []),  # Already an array from backend
                                    experience=int(float(item.get('experience', 0))),
                                    rating=float(item.get('rating', 0)),
                                    hourly_rate=float(item.get('hourly_rate', 0)),
                                    profile_url=str(item.get('profile_url', '')),
                                    availability=bool(item.get('availability', False)),
                                    total_sales=int(float(item.get('total_sales', 0))),
                                    desc=str(item.get('desc', '')),
                                )
                                users.append(user)
                                logger.debug(f"Processed user: {user.name}")
                            except Exception as e:
                                logger.warning(f"Error processing user data: {e}")
                                continue

                        if not users:
                            logger.warning("No valid users found in response")
                            if Config.DEBUG:
                                return self._get_mock_users()
                            raise Exception("No valid users found")

                        return users

                except aiohttp.ClientError as e:
                    logger.error(f"API connection error: {str(e)}")
                    if Config.DEBUG:
                        return self._get_mock_users()
                    raise

        except Exception as e:
            logger.error(f"Error in load_users: {str(e)}")
            if Config.DEBUG:
                return self._get_mock_users()
            raise

    def _parse_skills(self, skills: Union[str, List, None]) -> List[str]:
        """Parse skills from various input formats"""
        if isinstance(skills, str):
            return [s.strip() for s in skills.split(',') if s.strip()]
        elif isinstance(skills, list):
            return [str(s).strip() for s in skills if str(s).strip()]
        return []

    def _get_mock_users(self) -> List[user]:
        """Provide mock user data for testing/development"""
        logger.info("Using mock user data")
        mock_data = [
            user(
                id="1",
                username="john_dev",
                name="John Developer",
                job_title="Full Stack Developer",
                skills=["Python", "JavaScript", "React"],
                experience=5,
                rating=4.8,
                hourly_rate=50.0,
                profile_url="http://example.com/john",
                availability=True,
                total_sales=20,
                desc="an experienced Full Stack Developer with over 5 years in the field. He specializes in Python, JavaScript, and React, enabling him to build robust, scalable, and interactive web applications. With a stellar rating of 4.8/5 and an hourly rate of $50, John has successfully completed 20 projects, demonstrating his reliability and expertise. His strong skill set makes him an ideal candidate for complex and dynamic web development projects."
            ),
            user(
                id="2",
                username="jane_designer",
                name="Jane Designer",
                job_title="UI/UX Designer",
                skills=["UI Design", "UX Research", "Figma", "Adobe XD"],
                experience=3,
                rating=4.9,
                hourly_rate=45.0,
                profile_url="http://example.com/jane",
                availability=True,
                total_sales=15,
                desc="a talented UI/UX Designer with 3 years of experience in creating intuitive and visually appealing designs. Jane excels in UI Design, UX Research, Figma, and Adobe XD, allowing her to deliver engaging user experiences. With a remarkable rating of 4.9/5 and an hourly rate of $45, Jane has successfully completed 15 projects, showcasing her creativity and attention to detail. Her expertise in UI/UX design makes her an excellent choice for innovative design projects."
            )
        ]
        return mock_data

    async def initialize_matching_engine(self):
        """Initialize a Upwork-customized matching engine with loaded users"""
        try:
            # Load users if not already loaded
            if not self.users:
                self.users = await self.load_users()
            
            if not self.users:
                logger.error("No users available for matching engine")
                raise Exception("No users available")

            collaborative_model = self.customize_matching_engine()
            self.matching_engine = MatchingEngine(
                users=self.users,
                projects=[],
                skill_extractor=self.skill_extractor,
                collaborative_model=collaborative_model,
            )
            self.matching_engine.train_models()
            logger.info("Matching engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing matching engine: {e}")
            raise
        
    def customize_matching_engine(self):
        """
        Customizes the collaborative model for Upwork to include total_jobs and success_rate.
        """
        class UpworkCollaborativeModel(CollaborativeModel):
            def train(self, project_data: List[Dict], user_data: List[user]):
                """
                Train the collaborative model using total_jobs and success_rate for Upwork.
                """
                self.user_data = user_data
                self.project_data = project_data

                num_users = len(user_data)
                if num_users == 0:
                    self.interaction_matrix = np.zeros((num_users, 2))
                    return

                total_jobs = np.array([user.total_sales for user in user_data])
                success_rates = np.array([user.rating for user in user_data])

                total_jobs_norm = (total_jobs - total_jobs.min()) / (total_jobs.max() - total_jobs.min())
                success_rates_norm = success_rates / 100.0

                self.interaction_matrix = np.column_stack((total_jobs_norm, success_rates_norm))

            def predict(self, project_description: str, project_skills: List[str]) -> List[float]:
                """
                Predict match scores based on collaborative metrics.
                """
                if self.interaction_matrix is None or self.interaction_matrix.size == 0:
                    logger.warning("Interaction matrix is empty. Returning zero scores.")
                    return [0.0] * len(self.user_data)

                scores = np.nanmean(self.interaction_matrix, axis=1)
                return np.nan_to_num(scores).tolist()

        return UpworkCollaborativeModel()

    def adjust_weights_for_project(self, project: Project) -> Dict[str, float]:
        """
        Dynamically adjust weights based on project complexity and Upwork-specific factors.
        """
        base_weights = self.custom_weights.copy()
        if project.complexity == 'high':
            base_weights['content'] += 0.1
            base_weights['collaborative'] += 0.1
            base_weights['top_rated'] += 0.1
        elif project.complexity == 'low':
            base_weights['hourly_rate'] = 0.2
        return base_weights

    def filter_users(self, project: Project, matches: List[Dict]) -> List[Dict]:
        """
        Filter users based on Upwork-specific constraints.
        """
        filtered_matches = []
        for match in matches:
            user = match['user']
            
            # Upwork-specific hard constraints
            if user.hourly_rate < project.budget_range[0] or user.hourly_rate > project.budget_range[1]:
                logger.debug(f"Excluded {user.username}: Hourly rate ${user.hourly_rate} out of budget.")
                continue
            
            # Prioritize top-rated users for critical projects
            if project.complexity == 'high' and not user.availability:
                logger.debug(f"Excluded {user.username}: Not top-rated for high-complexity project.")
                continue
            
            # Refine skill matching and check overlap
            overlap_count = self.matching_engine.refine_skill_matching(project.required_skills, user.skills)
            if overlap_count < 2:  # Require at least 2 overlapping or similar skills
                logger.debug(f"Excluded {user.username}: Insufficient skill overlap ({overlap_count} matching skills).")
                continue
            
            # Passed all filters
            filtered_matches.append(match)
        return filtered_matches

    async def find_top_matches(self, project: Project, top_n: int = 5):
        """Find top matching users for a project"""
        try:
            # Ensure matching engine is initialized
            if not self.matching_engine:
                await self.initialize_matching_engine()

            self.custom_weights = self.adjust_weights_for_project(project)
            all_matches = self.matching_engine.match_users(project, weights=self.custom_weights)
            filtered_matches = self.filter_users(project, all_matches)

            top_matches = filtered_matches[:top_n]
            logger.info(f"Found {len(top_matches)} top matches")
            return top_matches
        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            raise
    def run_upwork_matching(self):
        """
        Main workflow for Upwork user matching.
        """
        try:
            self.load_users()
            project = self.collect_project_details()
            top_matches = self.find_top_matches(project)

            for match in top_matches:
                user = match['user']
                print(f"\nCandidate: {user.name}")
                print(f"Match Score: {match['combined_score']:.2f}")
                print(f"Job Title: {user.job_title}")
                print(f"Skills: {', '.join(user.skills)}")
                print(f"Hourly Rate: {user.hourly_rate}")
                print(f"Total Jobs: {user.total_sales}")
                print(f"Success Rate: {user.rating}%")

                if input("Interview this user? (yes/no): ").strip().lower() == 'yes':
                    interview_results = self.interview_user(user, project)
                    print(f"Interview Results: {interview_results}")

                    if input("Hire this user? (yes/no): ").strip().lower() == 'yes':
                        logger.info(f"Hired user: {user.username}")
                        break
        except Exception as e:
            logger.error(f"Upwork matching process error: {e}")

    async def interview_user(self, user: user, project: Project) -> Dict[str, Any]:
        """
        Conduct real-time interview with a user through Upwork's API
        """
        try:
            # Generate interview questions
            questions = self.skill_extractor.generate_ai_interview_questions(
                project.description,
                user.skills
            )

            # Create interview session
            interview_session = {
                'user_id': user.id,
                'project_id': project.id,
                'questions': questions,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'pending'
            }

            # In a real implementation, you would:
            # 1. Send notification to user through Upwork's API
            # 2. Create a webhook to receive user's responses
            # 3. Handle real-time updates

            return {
                'interview_id': str(uuid.uuid4()),
                'status': 'initiated',
                'questions': questions,
                'user': {
                    'id': user.id,
                    'name': user.name,
                    'profile_url': user.profile_url
                }
            }

        except Exception as e:
            logger.error(f"Interview error: {e}")
            raise

    async def get_user_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get real-time user availability status
        """
        try:
            # In production, replace with actual Upwork API call
            return {
                'user_id': user_id,
                'online_status': 'available',
                'last_active': datetime.utcnow().isoformat(),
                'response_time': '2h'
            }
        except Exception as e:
            logger.error(f"Error getting user status: {e}")
            raise

    async def send_interview_invitation(self, user_id: str, project_id: str) -> Dict[str, Any]:
        """
        Send interview invitation to user through Upwork
        """
        try:
            # In production, implement actual Upwork API call
            invitation = {
                'user_id': user_id,
                'project_id': project_id,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'sent'
            }
            return invitation
        except Exception as e:
            logger.error(f"Error sending interview invitation: {e}")
            raise

    def evaluate_answers(self, answers: Dict[str, str]) -> Dict[str, Any]:
        """
        Enhanced answer evaluation with AI scoring
        """
        try:
            # Use AI model to evaluate answers
            evaluations = []
            total_score = 0

            for question, answer in answers.items():
                # Evaluate answer quality
                relevance_score = self._calculate_relevance(question, answer)
                completeness_score = self._calculate_completeness(answer)
                expertise_score = self._calculate_expertise_level(answer)
                
                score = (relevance_score + completeness_score + expertise_score) / 3
                total_score += score
                
                evaluations.append({
                    'question': question,
                    'answer': answer,
                    'scores': {
                        'relevance': relevance_score,
                        'completeness': completeness_score,
                        'expertise': expertise_score
                    }
                })

            return {
                'overall_score': total_score / len(answers),
                'detailed_evaluation': evaluations,
                'recommendation': self._generate_recommendation(total_score / len(answers))
            }
        except Exception as e:
            logger.error(f"Error evaluating answers: {e}")
            raise

    def _calculate_relevance(self, question: str, answer: str) -> float:
        # Implement relevance scoring logic
        return 0.8  # Placeholder

    def _calculate_completeness(self, answer: str) -> float:
        # Implement completeness scoring logic
        return 0.7  # Placeholder

    def _calculate_expertise_level(self, answer: str) -> float:
        # Implement expertise scoring logic
        return 0.9  # Placeholder

    def _generate_recommendation(self, score: float) -> str:
        if score >= 0.8:
            return "Highly Recommended"
        elif score >= 0.6:
            return "Recommended"
        else:
            return "Consider Other Candidates"

    def collect_project_details(self) -> Project:
        try:
            logger.info("Starting project details collection")

            description = input("Enter Project Description: ").strip()
            inferred_skills = self.skill_extractor.extract_skills(description)
            logger.info(f"AI-extracted skills: {inferred_skills}")

            manual_skills_input = input("Add additional skills (comma-separated, or press Enter to skip): ").strip()
            manual_skills = [skill.strip() for skill in manual_skills_input.split(',')] if manual_skills_input else []
            required_skills = list(set(inferred_skills + manual_skills))

            min_budget = self._get_valid_input("Minimum Budget ($): ", float)
            max_budget = self._get_valid_input("Maximum Budget ($): ", float, lambda x: x >= min_budget)

            complexity = self._get_choice_input("Project Complexity (low/medium/high): ", ['low', 'medium', 'high'])
            timeline = self._get_valid_input("Project Timeline (days): ", int, lambda x: x > 0)

            project = Project(
                id=str(uuid.uuid4()),
                description=description,
                required_skills=required_skills,
                budget_range=(min_budget, max_budget),
                complexity=complexity,
                timeline=timeline,
            )
            logger.info("Project details collected successfully")
            return project
        except Exception as e:
            logger.error(f"Error collecting project details: {e}")
            raise

    @staticmethod
    def _get_valid_input(prompt, input_type, condition=lambda x: True):
        while True:
            try:
                value = input_type(input(prompt))
                if not condition(value):
                    raise ValueError("Invalid input.")
                return value
            except ValueError:
                print("Invalid input. Please try again.")

    @staticmethod
    def _get_choice_input(prompt, choices):
        while True:
            choice = input(prompt).strip().lower()
            if choice in choices:
                return choice
            print(f"Invalid choice. Please select from {', '.join(choices)}.")
