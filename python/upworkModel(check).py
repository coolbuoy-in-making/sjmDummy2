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
    freelancer, 
    Project, 
    MatchingEngine,
    CollaborativeModel
)

# Add missing dict method to freelancer class in sjm.py
def dict(self) -> Dict[str, Any]:
    return {
        'id': self.id,
        'freelancername': self.freelancername,
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
        self.freelancers = None
        self.matching_engine = None
        self.custom_weights = {
            'content': 0.25,
            'collaborative': 0.25,
            'experience': 0.2,
            'rating': 0.1,
            'hourly_rate': 0.1,
            'top_rated': 0.0,
            'job_title': 0.1  # Add job title weight
        }
        self.api_headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.job_title_keywords = {
            'developer': ['developer', 'programmer', 'software engineer', 'coder', 'full stack'],
            'designer': ['designer', 'ui/ux', 'graphic designer', 'web designer'],
            'writer': ['writer', 'content creator', 'copywriter', 'editor'],
            'marketing': ['marketer', 'digital marketer', 'seo specialist', 'growth hacker'],
            'data': ['data scientist', 'data analyst', 'ml engineer', 'ai developer'],
            'project manager': ['project manager', 'product manager', 'scrum master', 'agile coach']
        }
        # Add greetings to ignore
        self.greetings = {'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'}
        
        # Initialize collections for database-driven keyword matching
        self.skill_frequencies = {}
        self.job_title_frequencies = {}
        self.skill_cooccurrence = {}

    async def load_freelancers(self) -> List[freelancer]:
        """Load freelancers from backend API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f'{self.api_url}/api/freelancers'  # Use freelancers endpoint
                logger.info(f"Fetching freelancers from {url}")
                
                async with session.get(
                    url,
                    headers=self.api_headers,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Successfully loaded {len(data)} freelancers")
                        
                        return [
                            freelancer(
                                id=str(item['id']),
                                freelancername=item.get('freelancername', ''),
                                name=item['name'],
                                job_title=item.get('jobTitle', ''),
                                skills=item.get('skills', []),
                                experience=int(item.get('yearsOfExperience', 0)),
                                rating=float(item.get('rating', 0)),
                                hourly_rate=float(item.get('hourlyRate', 0)),
                                profile_url=item.get('profileUrl', ''),
                                availability=bool(item.get('availability', False)),
                                total_sales=int(item.get('totalJobs', 0)),
                                desc=item.get('desc', '')
                            )
                            for item in data
                        ]
                    else:
                        raise Exception(f"API Error: {response.status}")
        except Exception as e:
            logger.error(f"Error loading freelancers: {e}")
            if Config.DEBUG:
                return self._get_mock_freelancers()
            raise

    def _parse_skills(self, skills: Union[str, List, None]) -> List[str]:
        """Parse skills from various input formats"""
        if isinstance(skills, str):
            return [s.strip() for s in skills.split(',') if s.strip()]
        elif isinstance(skills, list):
            return [str(s).strip() for s in skills if str(s).strip()]
        return []

    def _get_mock_freelancers(self) -> List[freelancer]:
        """Provide mock freelancer data for testing/development"""
        logger.info("Using mock freelancer data")
        mock_data = [
            freelancer(
                id="1",
                freelancername="john_dev",
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
            freelancer(
                id="2",
                freelancername="jane_designer",
                name="Jane Designer",
                job_title="UI/UX Designer",
                skills=["UI Design", "UX Research", "Figma", "Adobe XD"],
                experience=3,
                rating=4.9,
                hourly_rate=45.0,
                profile_url="http://example.com/jane",
                availability=True,
                total_sales=15,
                desc="a talented UI/UX Designer with 3 years of experience in creating intuitive and visually appealing designs. Jane excels in UI Design, UX Research, Figma, and Adobe XD, allowing her to deliver engaging freelancer experiences. With a remarkable rating of 4.9/5 and an hourly rate of $45, Jane has successfully completed 15 projects, showcasing her creativity and attention to detail. Her expertise in UI/UX design makes her an excellent choice for innovative design projects."
            )
        ]
        return mock_data

    async def initialize_matching_engine(self):
        """Initialize a Upwork-customized matching engine with loaded freelancers"""
        try:
            # Load freelancers if not already loaded
            if not self.freelancers:
                self.freelancers = await self.load_freelancers()
            
            if not self.freelancers:
                logger.error("No freelancers available for matching engine")
                raise Exception("No freelancers available")

            # Initialize skill extractor with database terms
            self.skill_extractor.load_keywords_from_database(self.freelancers)
            
            # Build frequency and co-occurrence data
            self._build_skill_statistics()
            
            # Initialize matching engine
            collaborative_model = self.customize_matching_engine()
            self.matching_engine = MatchingEngine(
                freelancers=self.freelancers,
                projects=[],
                skill_extractor=self.skill_extractor,
                collaborative_model=collaborative_model,
            )
            self.matching_engine.train_models()
            logger.info("Matching engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing matching engine: {e}")
            raise

    def _build_skill_statistics(self):
        """Build statistics about skills and job titles from actual data"""
        for freelancer in self.freelancers:
            # Count job title frequencies
            title = freelancer.job_title.lower()
            self.job_title_frequencies[title] = self.job_title_frequencies.get(title, 0) + 1
            
            # Count skill frequencies
            for skill in freelancer.skills:
                skill_lower = skill.lower()
                self.skill_frequencies[skill_lower] = self.skill_frequencies.get(skill_lower, 0) + 1
                
                # Build skill co-occurrence matrix
                for other_skill in freelancer.skills:
                    if skill_lower != other_skill.lower():
                        if skill_lower not in self.skill_cooccurrence:
                            self.skill_cooccurrence[skill_lower] = {}
                        self.skill_cooccurrence[skill_lower][other_skill.lower()] = \
                            self.skill_cooccurrence[skill_lower].get(other_skill.lower(), 0) + 1

    def customize_matching_engine(self):
        """
        Customizes the collaborative model for Upwork to include total_jobs and success_rate.
        """
        class UpworkCollaborativeModel(CollaborativeModel):
            def train(self, project_data: List[Dict], freelancer_data: List[freelancer]):
                """
                Train the collaborative model using total_jobs and success_rate for Upwork.
                """
                self.freelancer_data = freelancer_data
                self.project_data = project_data

                num_freelancers = len(freelancer_data)
                if num_freelancers == 0:
                    self.interaction_matrix = np.zeros((num_freelancers, 2))
                    return

                total_jobs = np.array([freelancer.total_sales for freelancer in freelancer_data])
                success_rates = np.array([freelancer.rating for freelancer in freelancer_data])

                total_jobs_norm = (total_jobs - total_jobs.min()) / (total_jobs.max() - total_jobs.min())
                success_rates_norm = success_rates / 100.0

                self.interaction_matrix = np.column_stack((total_jobs_norm, success_rates_norm))

            def predict(self, project_description: str, project_skills: List[str]) -> List[float]:
                """
                Predict match scores based on collaborative metrics.
                """
                if self.interaction_matrix is None or self.interaction_matrix.size == 0:
                    logger.warning("Interaction matrix is empty. Returning zero scores.")
                    return [0.0] * len(self.freelancer_data)

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
            base_weights['job_title'] += 0.1  # Increase job title importance for complex projects
        elif project.complexity == 'low':
            base_weights['hourly_rate'] = 0.2
            base_weights['job_title'] = 0.15  # Adjust for low complexity
        return base_weights

    def filter_freelancers(self, project: Project, matches: List[Dict]) -> List[Dict]:
        """Filter freelancers based on Upwork-specific constraints."""
        filtered_matches = []
        logger.info(f"\nFiltering {len(matches)} potential matches:")
        
        for match in matches:
            freelancer = match['freelancer']
            logger.info(f"\nEvaluating {freelancer.name}:")
            
            # Budget filter
            if freelancer.hourly_rate < project.budget_range[0] or freelancer.hourly_rate > project.budget_range[1]:
                logger.info(f"[X] Excluded: Hourly rate ${freelancer.hourly_rate} outside budget range ${project.budget_range[0]}-${project.budget_range[1]}")
                continue
            logger.info(f"[✓] Budget check passed: ${freelancer.hourly_rate}/hr")
            
            # Availability check for high complexity
            if project.complexity == 'high' and not freelancer.availability:
                logger.info("[X] Excluded: Not available for high-complexity project")
                continue
            logger.info(f"[✓] Availability check passed")
            
            # Skill matching
            overlap_count = self.matching_engine.refine_skill_matching(project.required_skills, freelancer.skills)
            
            if project.complexity == 'high' and not freelancer.availability:
                if overlap_count < 2:
                    logger.info(f"[X] Excluded: Only {overlap_count} matching skills")
                    continue
                logger.info(f"[✓] Skill match: {overlap_count} overlapping skills")
            
            # Add match score details
            match['skill_overlap'] = overlap_count
            match['experience_score'] = freelancer.experience / 10
            match['rating_score'] = freelancer.rating / 5
            
            filtered_matches.append(match)
            logger.info(f"-> Added to matches with score: {match['combined_score']:.2f}")
        
        if not filtered_matches:
            suggestions = self.generate_search_suggestions(project)
            logger.info("\nNo matches found. Generated suggestions:")
            for suggestion in suggestions:
                logger.info(f"- {suggestion}")
            
        return filtered_matches

    def generate_search_suggestions(self, project: Project) -> List[Dict]:
        """Generate helpful suggestions when no matches are found"""
        suggestions = []
        
        # Budget suggestions
        if project.budget_range[1] < 50:
            suggestions.append({
                'type': 'budget',
                'message': "Consider increasing your budget range. Quality freelancers typically charge higher rates.",
                'action': f"Try budget range: ${project.budget_range[1]}-${project.budget_range[1] * 2}/hr"
            })
        
        # Skills suggestions
        if len(project.required_skills) > 4:
            core_skills = project.required_skills[:3]
            suggestions.append({
                'type': 'skills',
                'message': "Try focusing on core skills first",
                'action': f"Search with main skills: {', '.join(core_skills)}"
            })
        
        # Experience level suggestions
        if project.complexity == 'high':
            suggestions.append({
                'type': 'experience',
                'message': "For complex projects, consider being more flexible with availability",
                'action': "Include freelancers who are currently busy but highly qualified"
            })
        
        # Similar skills suggestions
        similar_skills = self.find_similar_skills(project.required_skills)
        if similar_skills:
            suggestions.append({
                'type': 'alternative_skills',
                'message': "Consider these related skills",
                'action': f"Try including: {', '.join(similar_skills)}"
            })
        
        return suggestions

    def find_similar_skills(self, skills: List[str]) -> List[str]:
        """Find similar or related skills"""
        skill_map = {
            'react': ['reactjs', 'react.js', 'react native'],
            'nodejs': ['node.js', 'node', 'express.js'],
            'python': ['django', 'flask', 'fastapi'],
            'ui': ['freelancer interface', 'ux design', 'web design'],
            'aws': ['cloud computing', 'devops', 'azure'],
            # Add more mappings as needed
        }
        
        similar = set()
        for skill in skills:
            skill_lower = skill.lower()
            for key, values in skill_map.items():
                if skill_lower in [key] + values:
                    similar.update(values)
        
        return list(similar - set(skills))

    async def find_top_matches(self, project: Project, top_n: int = 5):
        """Find top matching freelancers for a project"""
        try:
            if not self.matching_engine:
                await self.initialize_matching_engine()

            self.custom_weights = self.adjust_weights_for_project(project)
            all_matches = self.matching_engine.match_freelancers(
                project, 
                weights=self.custom_weights,
                job_title_matcher=self._calculate_job_title_match
            )
            filtered_matches = self.filter_freelancers(project, all_matches)

            # Format matches for frontend with enhanced job title relevance
            formatted_matches = [{
                'id': str(match['freelancer'].id),
                'name': match['freelancer'].name,
                'jobTitle': match['freelancer'].job_title,
                'skills': match['freelancer'].skills,
                'hourlyRate': float(match['freelancer'].hourly_rate),
                'rating': float(match['freelancer'].rating),
                'availability': bool(match['freelancer'].availability),
                'totalSales': int(match['freelancer'].total_sales),
                'desc': match['freelancer'].desc,
                'matchDetails': {
                    'skillMatch': {
                        'skills': project.required_skills,
                        'count': match.get('skill_overlap', 0)
                    },
                    'jobTitleMatch': {
                        'score': match.get('job_title_score', 0),
                        'relevance': 'High' if match.get('job_title_score', 0) > 0.8 
                                    else 'Medium' if match.get('job_title_score', 0) > 0.5 
                                    else 'Low'
                    },
                    'matchPercentage': round(float(match['combined_score'] * 100), 1),
                    'experienceScore': match.get('experience_score', 0),
                    'contentScore': match.get('content_score', 0),
                    'collaborativeScore': match.get('collaborative_score', 0)
                }
            } for match in filtered_matches[:top_n]]

            logger.info(f"Found {len(formatted_matches)} top matches")
            return formatted_matches

        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            raise

    def run_upwork_matching(self):
        """
        Main workflow for Upwork freelancer matching.
        """
        try:
            self.load_freelancers()
            project = self.collect_project_details()
            top_matches = self.find_top_matches(project)

            for match in top_matches:
                freelancer = match['freelancer']
                print(f"\nCandidate: {freelancer.name}")
                print(f"Match Score: {match['combined_score']:.2f}")
                print(f"Job Title: {freelancer.job_title}")
                print(f"Skills: {', '.join(freelancer.skills)}")
                print(f"Hourly Rate: {freelancer.hourly_rate}")
                print(f"Total Jobs: {freelancer.total_sales}")
                print(f"Success Rate: {freelancer.rating}%")

                if input("Interview this freelancer? (yes/no): ").strip().lower() == 'yes':
                    interview_results = self.interview_freelancer(freelancer, project)
                    print(f"Interview Results: {interview_results}")

                    if input("Hire this freelancer? (yes/no): ").strip().lower() == 'yes':
                        logger.info(f"Hired freelancer: {freelancer.freelancername}")
                        break
        except Exception as e:
            logger.error(f"Upwork matching process error: {e}")

    async def interview_freelancer(self, freelancer: freelancer, project: Project) -> Dict[str, Any]:
        """
        Conduct real-time interview with a freelancer through Upwork's API
        """
        try:
            # Generate interview questions
            questions = self.skill_extractor.generate_ai_interview_questions(
                project.description,
                freelancer.skills
            )

            # Create interview session
            interview_session = {
                'freelancer_id': freelancer.id,
                'project_id': project.id,
                'questions': questions,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'pending'
            }

            # In a real implementation, you would:
            # 1. Send notification to freelancer through Upwork's API
            # 2. Create a webhook to receive freelancer's responses
            # 3. Handle real-time updates

            return {
                'interview_id': str(uuid.uuid4()),
                'status': 'initiated',
                'questions': questions,
                'freelancer': {
                    'id': freelancer.id,
                    'name': freelancer.name,
                    'profile_url': freelancer.profile_url
                }
            }

        except Exception as e:
            logger.error(f"Interview error: {e}")
            raise

    async def get_freelancer_status(self, freelancer_id: str) -> Dict[str, Any]:
        """
        Get real-time freelancer availability status
        """
        try:
            # In production, replace with actual Upwork API call
            return {
                'freelancer_id': freelancer_id,
                'online_status': 'available',
                'last_active': datetime.utcnow().isoformat(),
                'response_time': '2h'
            }
        except Exception as e:
            logger.error(f"Error getting freelancer status: {e}")
            raise

    async def send_interview_invitation(self, freelancer_id: str, project_id: str) -> Dict[str, Any]:
        """
        Send interview invitation to freelancer through Upwork
        """
        try:
            # In production, implement actual Upwork API call
            invitation = {
                'freelancer_id': freelancer_id,
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

    def _calculate_job_title_match(self, project_description: str, freelancer_title: str) -> float:
        """Calculate job title relevance score"""
        try:
            # Normalize text
            description_lower = project_description.lower()
            title_lower = freelancer_title.lower()

            # Direct match bonus
            if any(keyword in title_lower for keyword in description_lower.split()):
                return 1.0

            # Category match
            for category_keywords in self.job_title_keywords.values():
                category_in_description = any(kw in description_lower for kw in category_keywords)
                category_in_title = any(kw in title_lower for kw in category_keywords)
                
                if category_in_description and category_in_title:
                    return 0.8

            # Partial match using word overlap
            desc_words = set(description_lower.split())
            title_words = set(title_lower.split())
            overlap = len(desc_words.intersection(title_words))
            
            if overlap > 0:
                return 0.5 + (0.1 * overlap)  # Bonus for each matching word

            return 0.0
        except Exception as e:
            logger.error(f"Error in job title matching: {e}")
            return 0.0

    async def analyze_job_title(self, title: str) -> Dict[str, Any]:
        """Enhanced job title analysis with accurate skill mapping"""
        try:
            # Check for greeting
            if title.lower().strip() in self.greetings:
                return {
                    'type': 'greeting',
                    'message': "Hi! I can help you find freelancers. What kind of skills or expertise are you looking for?"
                }

            # Split and analyze each word
            verification = self.skill_extractor.verify_keyword(title)
            
            # Log the verification results
            logger.debug(f"Keyword verification results: {verification}")
            
            if not verification['exists']:
                similar_terms = verification['similar_terms']
                popular_cats = self._get_popular_categories()
                
                return {
                    'type': 'no_match',
                    'message': f"I couldn't find any exact matches for '{title}'. Here are some related options:",
                    'suggestions': similar_terms,
                    'popular_categories': popular_cats,
                    'suggested_skills': self._get_related_skills(title.lower())
                }

            # Find matching freelancers based on verified skills and job titles
            matching_freelancers = []
            
            # Search by skills
            for skill in verification['skills']:
                skill_matches = [
                    f for f in self.freelancers 
                    if any(skill.lower() in s.lower() for s in f.skills)
                ]
                matching_freelancers.extend(skill_matches)
                
            # Search by job titles
            for job_title in verification['job_titles']:
                title_matches = [
                    f for f in self.freelancers 
                    if job_title.lower() in f.job_title.lower()
                ]
                matching_freelancers.extend(title_matches)

            # Remove duplicates while preserving order
            seen = set()
            matching_freelancers = [
                f for f in matching_freelancers 
                if not (f.id in seen or seen.add(f.id))
            ]

            if not matching_freelancers:
                return self._generate_no_matches_response(title)

            # Get skills from matching freelancers
            related_skills = set()
            skill_frequency = {}
            for f in matching_freelancers:
                for skill in f.skills:
                    if skill.lower() not in [k.lower() for k in verification['matches']]:
                        skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
                        related_skills.add(skill)

            # Calculate profile statistics
            stats = self._calculate_profile_statistics(matching_freelancers)

            return {
                'type': 'success',
                'keyword_type': verification['type'],
                'suggested_skills': list(related_skills)[:5],
                'avg_hourly_range': stats['hourly_range'],
                'sample_profiles': stats['sample_profiles'],
                'total_matches': len(matching_freelancers),
                'skill_requirements': stats['common_requirements']
            }

        except Exception as e:
            logger.error(f"Error analyzing job title: {e}", exc_info=True)
            return {
                'type': 'error',
                'message': f"Error analyzing request: {str(e)}",
                'error': str(e)
            }

    def _matches_keyword(self, freelancer: freelancer, keyword: str) -> bool:
        """Check if freelancer matches keyword in either title or skills"""
        keyword_lower = keyword.lower()
        return (
            keyword_lower in freelancer.job_title.lower() or
            any(keyword_lower in skill.lower() for skill in freelancer.skills)
        )

    def _get_related_skills(self, keyword: str) -> List[str]:
        """Get truly related skills from freelancers who have the input skill or job title"""
        try:
            related_skills = {}
            matching_freelancers = []
            keyword_lower = keyword.lower()

            # Find freelancers who have this skill or job title
            for freelancer in self.freelancers:
                # Check skills
                if any(keyword_lower in skill.lower() for skill in freelancer.skills):
                    matching_freelancers.append(freelancer)
                # Check job title
                elif keyword_lower in freelancer.job_title.lower():
                    matching_freelancers.append(freelancer)

            # Collect other skills from matching freelancers
            for freelancer in matching_freelancers:
                for skill in freelancer.skills:
                    skill_lower = skill.lower()
                    if skill_lower != keyword_lower:  # Don't include the search keyword
                        if skill_lower not in related_skills:
                            related_skills[skill_lower] = {
                                'name': skill,
                                'count': 1,
                                'freelancers': [freelancer.id]
                            }
                        else:
                            if freelancer.id not in related_skills[skill_lower]['freelancers']:
                                related_skills[skill_lower]['count'] += 1
                                related_skills[skill_lower]['freelancers'].append(freelancer.id)

            # Sort by frequency and get top skills
            sorted_skills = sorted(
                related_skills.values(),
                key=lambda x: x['count'],
                reverse=True
            )

            return [skill['name'] for skill in sorted_skills[:5]]

        except Exception as e:
            logger.error(f"Error getting related skills: {e}")
            return []

    def _get_similar_titles(self, query: str) -> List[str]:
        """Find similar job titles based on partial matches"""
        all_titles = set()
        all_skills = set()
        
        for f in self.freelancers:
            all_titles.add(f.job_title.lower())
            all_skills.update(s.lower() for s in f.skills)

        # Find partial matches
        similar_titles = [
            title for title in all_titles
            if any(word in title for word in query.split())
        ]

        similar_skills = [
            skill for skill in all_skills
            if any(word in skill for word in query.split())
        ]

        return list(set(similar_titles + similar_skills))[:5]

    def _get_popular_categories(self) -> List[str]:
        """Get popular job categories from database"""
        categories = {}
        
        for f in self.freelancers:
            category = f.job_title.split()[0]  # Use first word as category
            categories[category] = categories.get(category, 0) + 1
            
        return [k for k, v in sorted(categories.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)][:5]

    async def get_related_skills(self, skills: List[str]) -> List[str]:
        """Get truly related skills from freelancers with matching skills"""
        try:
            if not self.freelancers:
                return []

            # Convert input skills to lowercase for matching
            input_skills_lower = [s.lower() for s in skills]
            
            # Find freelancers with matching skills
            matching_freelancers = []
            for freelancer in self.freelancers:
                freelancer_skills_lower = [s.lower() for s in freelancer.skills]
                # Check for any matching skills
                if any(skill in freelancer_skills_lower for skill in input_skills_lower):
                    matching_freelancers.append(freelancer)

            if not matching_freelancers:
                # Try matching by job title keywords
                for freelancer in self.freelancers:
                    if any(skill in freelancer.job_title.lower() for skill in input_skills_lower):
                        matching_freelancers.append(freelancer)

            # Get related skills with frequency count
            related_skills = {}
            for freelancer in matching_freelancers:
                for skill in freelancer.skills:
                    skill_lower = skill.lower()
                    if skill_lower not in input_skills_lower:
                        if skill_lower not in related_skills:
                            related_skills[skill_lower] = {
                                'skill': skill,
                                'count': 1,
                                'freelancers': [freelancer.id]
                            }
                        else:
                            if freelancer.id not in related_skills[skill_lower]['freelancers']:
                                related_skills[skill_lower]['count'] += 1
                                related_skills[skill_lower]['freelancers'].append(freelancer.id)

            # Sort by frequency and return top related skills
            sorted_skills = sorted(
                related_skills.values(),
                key=lambda x: (x['count'], len(x['freelancers'])),
                reverse=True
            )

            return [item['skill'] for item in sorted_skills[:5]]

        except Exception as e:
            logger.error(f"Error getting related skills: {e}")
            return []

    def _calculate_profile_statistics(self, matching_freelancers: List[freelancer]) -> Dict[str, Any]:
        """Calculate statistics from matching freelancer profiles"""
        try:
            if not matching_freelancers:
                return {
                    'hourly_range': (0, 0),
                    'sample_profiles': [],
                    'common_requirements': []
                }

            # Calculate hourly range
            rates = [f.hourly_rate for f in matching_freelancers]
            hourly_range = (min(rates), max(rates))

            # Get common skills across profiles
            skill_freq = {}
            for f in matching_freelancers:
                for skill in f.skills:
                    skill_freq[skill] = skill_freq.get(skill, 0) + 1

            common_requirements = sorted(
                [(skill, count) for skill, count in skill_freq.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]

            # Format sample profiles
            sample_profiles = [{
                'id': f.id,
                'name': f.name,
                'skills': f.skills[:5],  # Top 5 skills
                'hourly_rate': f.hourly_rate,
                'job_title': f.job_title,
                'rating': f.rating,
                'total_sales': f.total_sales
            } for f in matching_freelancers[:3]]  # Top 3 profiles

            return {
                'hourly_range': hourly_range,
                'sample_profiles': sample_profiles,
                'common_requirements': [skill for skill, _ in common_requirements]
            }

        except Exception as e:
            logger.error(f"Error calculating profile statistics: {e}")
            return {
                'hourly_range': (0, 0),
                'sample_profiles': [],
                'common_requirements': []
            }

    def _generate_no_matches_response(self, title: str) -> Dict[str, Any]:
        """Generate response when no matches are found"""
        return {
            'type': 'no_match',
            'message': f"I couldn't find any exact matches for '{title}', but here are some similar options:",
            'suggestions': self._get_similar_titles(title),
            'popular_categories': self._get_popular_categories(),
            'suggested_skills': self._get_related_skills(title)
        }
