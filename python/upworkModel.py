import logging
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any, Union, Tuple
from datetime import datetime
import uuid
import json

from sjm import (
    SkillsExtract, 
    Freelancer, 
    Project, 
    MatchingEngine,
    CollaborativeModel
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('upwork_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UpworkMatchingEngine(MatchingEngine):
    """Upwork-specific extension of base MatchingEngine"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_weights = {
            'skills': 0.45,
            'experience': 0.20,
            'rating': 0.15,
            'job_title': 0.10,
            'availability': 0.10
        }

    def match_Freelancers(self, project: Project, **kwargs):
        """Override with Upwork-specific matching logic"""
        try:
            weights = kwargs.get('weights', self.custom_weights)
            page = kwargs.get('page', 1)
            
            project_skills = [s.lower() for s in project.required_skills]
            logger.debug(f"Processing matches for skills: {project_skills}")

            all_matches = []
            
            for freelancer in self.Freelancers:
                try:
                    # Only consider available freelancers
                    if not freelancer.availability:
                        continue

                    freelancer_skills = [s.lower() for s in freelancer.skills]
                    
                    # Calculate matches
                    matched_skills = set()
                    
                    # Exact and partial skill matches
                    for ps in project_skills:
                        for fs in freelancer_skills:
                            if ps == fs:
                                matched_skills.add(fs)
                            elif ps in fs or fs in ps:  # Partial matches
                                matched_skills.add(fs)
                    
                    # Calculate detailed scores
                    skill_score = len(matched_skills) / len(project_skills) if project_skills else 0
                    experience_score = min(freelancer.experience / 10.0, 1.0)
                    rating_score = freelancer.rating / 5.0 if freelancer.rating else 0

                    # Calculate weighted combined score
                    combined_score = (
                        weights['skills'] * skill_score +
                        weights['experience'] * experience_score +
                        weights['rating'] * rating_score
                    )

                    # Calculate match percentage (0-100)
                    match_percentage = round(combined_score * 100, 1)
                    
                    match = {
                        'freelancer': {
                            **freelancer.dict(),
                            'rating': round(float(freelancer.rating), 2),
                            'hourlyRate': round(float(freelancer.hourly_rate), 2),
                        },
                        'matchDetails': {
                            'skillMatch': {
                                'skills': list(matched_skills),
                                'count': len(matched_skills)
                            },
                            'matchPercentage': match_percentage,
                            'experienceScore': round(experience_score * 100, 1),
                            'ratingScore': round(rating_score * 100, 1),
                            'skillScore': round(skill_score * 100, 1)
                        },
                        'matched_skills': list(matched_skills),
                        'skill_overlap': len(matched_skills),
                        'combined_score': combined_score
                    }
                    
                    if match['combined_score'] > 0:
                        all_matches.append(match)
                        
                except Exception as e:
                    logger.error(f"Error processing freelancer {freelancer.id}: {e}")
                    continue

            # Sort by combined score and take top 5
            all_matches.sort(key=lambda x: (-x['combined_score'], -x['skill_overlap']))
            top_matches = all_matches[:5]
            
            return {
                'freelancers': [
                    {
                        **m['freelancer'],
                        'matchDetails': m['matchDetails']  # Include match details in freelancer data
                    } for m in top_matches
                ],
                'total': len(top_matches),
                'page': page,
                'hasMore': False
            }

        except Exception as e:
            logger.error(f"Error in match_Freelancers: {e}")
            raise

    def _skill_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity between two skills"""
        # Direct match
        if skill1 == skill2:
            return 1.0
            
        # Substring match
        if skill1 in skill2 or skill2 in skill1:
            return 0.9
            
        # Use base class's similar method for other cases
        return self.similar(skill1, skill2)

class UpworkCollaborativeModel(CollaborativeModel):
    """Upwork-specific collaborative filtering model"""
    
    def train(self, project_data: List[Dict], freelancer_data: List[Freelancer]):
        """Customize training for Upwork metrics"""
        self.freelancer_data = freelancer_data
        
        # Add Upwork-specific metrics
        for freelancer in freelancer_data:
            freelancer.success_rate = freelancer.rating
            freelancer.job_completion = freelancer.total_sales
            
        super().train(project_data, freelancer_data)

class UpworkIntegrationModel:
    """Main class for Upwork integration"""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.Freelancers = []
        self.skill_extractor = SkillsExtract()
        self.matching_engine = None
        self.api_headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Add greetings set
        self.greetings = {
            'hi', 'hello', 'hey', 'greetings', 'good morning', 
            'good afternoon', 'good evening', 'hi there', 'hello there', 'how are you doing?'
        }
        
        # Add common job categories
        self.job_categories = {
            'development': ['developer', 'programmer', 'engineer', 'coder'],
            'design': ['designer', 'artist', 'creative'],
            'writing': ['writer', 'editor', 'content'],
            'marketing': ['marketer', 'seo', 'social media'],
            'data': ['data scientist', 'analyst', 'data engineer'],
            'project management': ['project manager', 'product manager', 'scrum master']
        }

    async def initialize(self) -> bool:
        """Initialize the model"""
        try:
            success = await self.load_Freelancers()
            if success:
                self.matching_engine = UpworkMatchingEngine(
                    Freelancers=self.Freelancers,
                    projects=[],
                    skill_extractor=self.skill_extractor,
                    collaborative_model=UpworkCollaborativeModel()
                )
                logger.info("UpworkIntegrationModel initialized successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False

    async def load_Freelancers(self) -> bool:
        """Load Freelancers from Upwork API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'{self.api_url}/api/freelancers',
                    headers=self.api_headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.Freelancers = [
                            Freelancer(**self._map_freelancer_data(item))
                            for item in data
                        ]
                        return bool(self.Freelancers)
            return False
        except Exception as e:
            logger.error(f"Error loading Freelancers: {e}")
            return False

    def _map_freelancer_data(self, data: Dict) -> Dict:
        """Map Upwork API data to Freelancer model"""
        return {
            'id': str(data.get('id')),
            'name': data.get('name', ''),
            'username': data.get('username', ''),
            'job_title': data.get('jobTitle', ''),
            'skills': data.get('skills', []),
            'hourly_rate': float(data.get('hourlyRate', 0)),
            'rating': float(data.get('rating', 0)),
            'profile_url': data.get('profileUrl', ''),
            'availability': bool(data.get('availability', False)),
            'total_sales': int(data.get('totalSales', 0)),
            'experience': int(data.get('experience', 0)),
            'desc': data.get('desc', '')
        }

    async def find_matches(self, project: Project, page: int = 1) -> Dict[str, Any]:
        """Find matching Freelancers for a project"""
        try:
            if not self.matching_engine:
                await self.initialize()

            result = self.matching_engine.match_Freelancers(project, page=page)
            
            return {
                'success': True,
                'response': {
                    'type': 'freelancerList',
                    'text': f"Found {result['total']} matching freelancers",
                    'freelancers': result['freelancers'],
                    'hasMore': result['hasMore'],
                    'currentPage': result['page']
                }
            }
            
        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            return {'success': False, 'error': str(e)}

    async def handle_chat_request(self, message: str, project_details: Optional[Dict] = None, page: int = 1) -> Dict:
        """Enhanced chat request handler with greeting and suggestion support"""
        try:
            if not project_details:
                message_lower = message.lower().strip()
                
                # Handle greetings
                if message_lower in self.greetings:
                    return {
                        'success': True,
                        'response': {
                            'type': 'greeting',
                            'text': "Hello! I can help you find the right freelancers. What skills or expertise are you looking for?",
                            'data': {
                                'suggestedCategories': list(self.job_categories.keys()),
                                'popularSkills': await self._get_popular_skills(),
                                'hourlyRange': [44, 150]  # Default range
                            }
                        }
                    }

                # Extract skills from message
                detected_skills = self.skill_extractor.extract_skills(message)
                
                # If no skills detected, search in job titles and skills
                if not detected_skills:
                    job_title_matches = await self._find_matching_job_titles(message_lower)
                    skill_matches = await self._find_matching_skills(message_lower)
                    
                    if not (job_title_matches or skill_matches):
                        popular_skills = await self._get_popular_skills()
                        return {
                            'success': True,
                            'response': {
                                'type': 'suggestions',
                                'text': f"How about i gave you these suggestions?:",
                                'data': {
                                    'suggestedJobTitles': await self._get_popular_job_titles(),
                                    'suggestedSkills': popular_skills,
                                    'categories': list(self.job_categories.keys()),
                                    'message': "Try these popular categories and skills:",
                                    'hourlyRange': [44, 150]  # Default range
                                }
                            }
                        }
                    
                    # Use found matches as detected skills
                    detected_skills = list(set(job_title_matches + skill_matches))

                # Get additional data for detected skills
                suggested_skills = await self.get_suggested_skills(detected_skills)
                hourly_range = await self.get_hourly_range(detected_skills)
                job_titles = await self.get_relevant_job_titles(detected_skills)
                
                return {
                    'success': True,
                    'response': {
                        'type': 'job_analysis',
                        'text': f"I found {len(detected_skills)} relevant skills for your request.",
                        'data': {
                            'detectedSkills': detected_skills or [],
                            'suggestedSkills': suggested_skills or [],
                            'hourlyRange': list(hourly_range),  # Convert tuple to list
                            'jobTitles': job_titles or []
                        }
                    }
                }
            
            # Handle project details case
            return await self._handle_project_details(project_details, page)
            
        except Exception as e:
            logger.error(f"Chat request error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': {
                    'type': 'error',
                    'data': {
                        'detectedSkills': [],
                        'suggestedSkills': [],
                        'hourlyRange': [44, 150],  # Default range
                        'jobTitles': []
                    }
                }
            }

    async def _get_popular_skills(self) -> List[str]:
        """Get most common skills from freelancers"""
        try:
            skill_count = {}
            for freelancer in self.Freelancers:
                for skill in freelancer.skills:
                    skill_count[skill] = skill_count.get(skill, 0) + 1
            
            # Sort by frequency and return top 5
            popular_skills = sorted(skill_count.items(), key=lambda x: x[1], reverse=True)
            return [skill for skill, _ in popular_skills[:5]]
        except Exception as e:
            logger.error(f"Error getting popular skills: {e}")
            return []

    async def _get_popular_job_titles(self) -> List[str]:
        """Get most common job titles"""
        try:
            title_count = {}
            for freelancer in self.Freelancers:
                title = freelancer.job_title
                title_count[title] = title_count.get(title, 0) + 1
            
            # Sort by frequency and return top 5
            popular_titles = sorted(title_count.items(), key=lambda x: x[1], reverse=True)
            return [title for title, _ in popular_titles[:5]]
        except Exception as e:
            logger.error(f"Error getting popular job titles: {e}")
            return []

    async def _find_matching_job_titles(self, query: str) -> List[str]:
        """Find skills from freelancers with matching job titles"""
        matching_skills = set()
        for freelancer in self.Freelancers:
            if query in freelancer.job_title.lower():
                matching_skills.update(freelancer.skills)
        return list(matching_skills)

    async def _find_matching_skills(self, query: str) -> List[str]:
        """Find skills that match the query"""
        matching_skills = set()
        for freelancer in self.Freelancers:
            for skill in freelancer.skills:
                if query in skill.lower():
                    matching_skills.add(skill)
        return list(matching_skills)

    async def get_suggested_skills(self, skills: List[str]) -> List[str]:
        """Get suggested complementary skills based on detected skills"""
        try:
            suggestions = set()
            for freelancer in self.Freelancers:
                if any(skill in freelancer.skills for skill in skills):
                    suggestions.update(s for s in freelancer.skills if s not in skills)
            return list(suggestions)[:5]  # Return top 5 suggestions
        except Exception as e:
            logger.error(f"Error getting suggested skills: {e}")
            return []

    async def get_hourly_range(self, skills: List[str]) -> Tuple[float, float]:
        """Get typical hourly range for given skills"""
        try:
            matching_rates = []
            for freelancer in self.Freelancers:
                if any(skill.lower() in [s.lower() for s in freelancer.skills] for skill in skills):
                    if freelancer.hourly_rate > 0:  # Only include valid rates
                        matching_rates.append(freelancer.hourly_rate)
            
            if matching_rates:
                return (
                    round(min(matching_rates), 2),
                    round(max(matching_rates), 2)
                )
            return (44.0, 150.0)  # Default range
        except Exception as e:
            logger.error(f"Error getting hourly range: {e}")
            return (44.0, 150.0)

    async def get_relevant_job_titles(self, skills: List[str]) -> List[str]:
        """Get relevant job titles based on skills"""
        try:
            titles = set()
            for freelancer in self.Freelancers:
                if any(skill in freelancer.skills for skill in skills):
                    titles.add(freelancer.job_title)
            return list(titles)[:5]  # Return top 5 titles
        except Exception as e:
            logger.error(f"Error getting job titles: {e}")
            return []

    def _parse_budget(self, budget_str: str) -> Tuple[float, float]:
        """Parse Upwork budget string format"""
        try:
            clean = budget_str.replace('$', '').replace('/hr', '').strip()
            min_val, max_val = map(float, clean.split('-'))
            return (min_val, max_val)
        except Exception:
            return (0.0, 100.0)

    def _prepare_project_form(self, skills: List[str]) -> Dict[str, Any]:
        """Prepare project form with Upwork-specific fields"""
        return {
            'success': True,
            'response': {
                'type': 'project_form',
                'detected_skills': skills,
                'fields': {
                    'skills': {'type': 'skill_list', 'value': skills},
                    'budget': {'type': 'budget_range', 'format': '$/hr'},
                    'complexity': {
                        'type': 'select',
                        'options': ['low', 'medium', 'high']
                    },
                    'timeline': {'type': 'number', 'unit': 'days'}
                }
            }
        }

    async def handle_interview_request(self, freelancer_id: str, project_id: str) -> Dict[str, Any]:
        """Handle Upwork-specific interview requests"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.api_url}/api/freelancers/interview-request',
                    headers=self.api_headers,
                    json={'freelancerId': freelancer_id, 'projectId': project_id}
                ) as response:
                    return await response.json()
        except Exception as e:
            logger.error(f"Interview request error: {e}")
            return {'success': False, 'error': str(e)}

    async def _handle_project_details(self, project_details: Dict[str, Any], page: int) -> Dict[str, Any]:
        """Process project details and find matches"""
        try:
            # Create project instance
            project = Project(
                id=str(uuid.uuid4()),
                description=project_details.get('description', ''),
                required_skills=project_details.get('skills', []),
                budget_range=self._parse_budget(project_details.get('budget', '')),
                complexity=project_details.get('complexity', 'medium'),
                timeline=int(project_details.get('timeline', 30))
            )
            
            # Find matches using matching engine
            result = await self.find_matches(project, page)
            
            # If no matches found, provide suggestions
            if not result.get('response', {}).get('freelancers'):
                return {
                    'success': True,
                    'response': {
                        'type': 'suggestions',
                        'text': "I couldn't find exact matches for your requirements. Here are some suggestions:",
                        'data': {
                            'suggestedSkills': await self._get_popular_skills(),
                            'suggestedJobTitles': await self._get_popular_job_titles(),
                            'categories': list(self.job_categories.keys()),
                            'message': "Try adjusting your requirements or consider these alternatives:",
                            'hourlyRange': [44, 150]  # Default range
                        }
                    }
                }
            
            # Add project requirements to the response
            result['response']['projectDetails'] = {
                'skills': project.required_skills,
                'budget': project_details.get('budget'),
                'complexity': project.complexity,
                'timeline': project.timeline
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling project details: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': {
                    'type': 'error',
                    'text': 'Sorry, I encountered an error processing your project details.',
                    'data': {
                        'detectedSkills': [],
                        'suggestedSkills': await self._get_popular_skills(),
                        'hourlyRange': [44, 150],
                        'jobTitles': await self._get_popular_job_titles()
                    }
                }
            }
