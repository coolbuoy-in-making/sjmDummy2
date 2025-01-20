from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List, Optional, Union, Any
import json
import re
import sys
import asyncio
from datetime import datetime
import uuid
import logging
from config import Config
from upworkModel import UpworkIntegrationModel
from sjm import MatchingEngine, Project
import asyncio
from functools import wraps
import nest_asyncio
nest_asyncio.apply()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('upwork_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize with API configuration
Config.validate()
model = None

def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(f(*args, **kwargs))
    return wrapped


async def initialize_model():
    """Initialize the UpworkIntegrationModel and load necessary data"""
    global model
    try:
        if not Config.API_KEY:
            raise ValueError("API_KEY is not configured")
            
        model = UpworkIntegrationModel(
            api_url=Config.API_URL,
            api_key=Config.API_KEY
        )
        # Initialize the matching engine (which includes loading users)
        await model.initialize_matching_engine()
        
        # Log available users
        users = model.users
        logger.info(f"\n{'='*50}\nLoaded {len(users)} users:")
        for i, f in enumerate(users[:10], 1):
            logger.info(f"\n{i}. {f.name} ({f.job_title})")
            logger.info(f"   Skills: {', '.join(f.skills)}")
            logger.info(f"   Rate: ${f.hourly_rate}/hr, Rating: {f.rating}%")
        if len(users) > 10:
            logger.info(f"\n... and {len(users)-10} more users")
        logger.info(f"{'='*50}\n")
        
        logger.info("Model initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}", exc_info=True)
        raise

def extract_project_details(message: str) -> Dict:
    """Extract project details from message using regex and NLP"""
    # Generate unique project ID
    project_id = str(uuid.uuid4())
    
    # Extract budget using regex
    budget_pattern = r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:[-–]\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?))?' 
    budget_match = re.search(budget_pattern, message)
    
    if budget_match:
        budget_min = float(budget_match.group(1).replace(',', ''))
        if budget_match.group(2):
            budget_max = float(budget_match.group(2).replace(',', ''))
        else:
            budget_max = budget_min * 1.5
    else:
        # Default budget range if not specified
        budget_min = 15
        budget_max = 150

    # Extract skills using model's skill extractor
    skills = model.skill_extractor.extract_skills(message)
    
    # Add common variations of skills
    expanded_skills = []
    for skill in skills:
        expanded_skills.append(skill)
        if skill.lower() == 'react':
            expanded_skills.extend(['ReactJS', 'React.js'])
        elif skill.lower() == 'node.js':
            expanded_skills.extend(['Node', 'NodeJS'])
    
    logger.info(f"Extracted skills: {expanded_skills}")
    logger.info(f"Budget range: ${budget_min}-${budget_max}")
    
    return {
        "id": project_id,  # Include ID in returned dictionary
        "description": message,
        "required_skills": expanded_skills,
        "budget_range": (budget_min, budget_max),
        "complexity": determine_project_complexity(message, expanded_skills),
        "timeline": extract_timeline(message)
    }

def determine_project_complexity(message: str, skills: List[str]) -> str:
    """
    Determine project complexity based on description and required skills.
    
    Args:
        message (str): Project description
        skills (List[str]): List of required skills
        
    Returns:
        str: Complexity level ('low', 'medium', or 'high')
    """
    # Count number of required skills
    skill_count = len(skills)
    
    # Look for complexity indicators in the message
    complex_indicators = ['complex', 'sophisticated', 'advanced', 'enterprise', 'scalable']
    simple_indicators = ['simple', 'basic', 'straightforward', 'easy']
    
    message_lower = message.lower()
    complex_count = sum(1 for indicator in complex_indicators if indicator in message_lower)
    simple_count = sum(1 for indicator in simple_indicators if indicator in message_lower)
    
    # Determine complexity based on indicators and skill count
    if skill_count >= 5 or complex_count >= 2:
        return 'high'
    elif skill_count <= 2 and simple_count >= 1:
        return 'low'
    else:
        return 'medium'

def extract_timeline(message: str) -> int:
    """
    Extract project timeline from message.
    
    Args:
        message (str): Project description
        
    Returns:
        int: Timeline in days (default: 30)
    """
    # Look for timeline indicators
    timeline_patterns = [
        r'(\d+)\s*(?:days?|d)',
        r'(\d+)\s*(?:weeks?|w)',
        r'(\d+)\s*(?:months?|m)'
    ]
    
    for pattern in timeline_patterns:
        match = re.search(pattern, message.lower())
        if match:
            value = int(match.group(1))
            if 'week' in pattern or 'w' in pattern:
                return value * 7
            elif 'month' in pattern or 'm' in pattern:
                return value * 30
            return value
            
    return 30  # Default timeline: 30 days

def extract_user_id(message: str) -> Optional[str]:
    """
    Extract user ID from message.
    
    Args:
        message (str): User message containing user reference
        
    Returns:
        Optional[str]: Extracted user ID or None if not found
    """
    # Look for user ID patterns
    id_patterns = [
        r'user[_-]?id[:\s]+([a-zA-Z0-9-]+)',
        r'interview\s+(?:user\s+)?([a-zA-Z0-9-]+)',
        r'#([a-zA-Z0-9-]+)'
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, message.lower())
        if match:
            return match.group(1)
            
    return None

class ConversationContext:
    def __init__(self):
        self.context = {}

    def update_context(self, user_id: str, context_data: Dict):
        if user_id not in self.context:
            self.context[user_id] = {}
        self.context[user_id].update(context_data)

    def get_context(self, user_id: str) -> Dict:
        return self.context.get(user_id, {})

conversation_context = ConversationContext()

@app.route('/ai/chat', methods=['POST'])
@async_route
async def chat() -> Union[Dict, tuple]:
    try:
        data = request.json
        message = data.get('message', '').lower()
        project_details = data.get('projectDetails')

        logger.info(f"\n{'='*50}\nProcessing chat request:")
        logger.info(f"Message: {message}")
        logger.info(f"Project details: {project_details}")

        if message == 'confirm_project_details' and project_details:
            logger.info("\nStarting freelancer search with confirmed details:")
            logger.info(f"Skills required: {project_details['skills']}")
            logger.info(f"Budget range: {project_details['budget']}")
            logger.info(f"Complexity: {project_details['complexity']}")
            
            # Create project from confirmed details
            project = Project(
                id=str(uuid.uuid4()),
                description=f"Project requiring skills in: {', '.join(project_details['skills'])}",
                required_skills=project_details['skills'],
                budget_range=parse_budget_range(project_details['budget']),
                complexity=project_details['complexity'],
                timeline=int(project_details['timeline'])
            )

            # Find matches with detailed logging
            matches = await model.find_top_matches(project)
            
            if not matches:
                logger.info("No matches found")
                return jsonify({
                    'success': True,
                    'response': {
                        'type': 'text',
                        'text': "I couldn't find any matches for your criteria. Try adjusting your requirements."
                    }
                })

            logger.info(f"\nFound {len(matches)} potential matches:")
            for i, match in enumerate(matches, 1):
                user = match['user']
                logger.info(f"\n{i}. {user.name} ({user.job_title})")
                logger.info(f"   Skills: {', '.join(user.skills)}")
                logger.info(f"   Match score: {match['combined_score']:.2f}")
                logger.info(f"   Skill overlap: {match.get('skill_overlap', 0)}")

            return jsonify({
                'success': True,
                'response': {
                    'type': 'freelancerList',
                    'text': f"I found {len(matches)} matching freelancers:",
                    'freelancers': format_user_matches(matches),
                    'matchingProcess': {
                        'steps': [
                            f"Analyzing {len(model.users)} available freelancers",
                            f"Found {len(matches)} potential matches",
                            f"Applied budget filter: ${project_details['budget']}",
                            f"Required skills: {', '.join(project_details['skills'])}"
                        ],
                        'searchStats': {
                            'totalFreelancers': len(model.users),
                            'matchesFound': len(matches),
                            'highMatches': sum(1 for m in matches if m['combined_score'] > 0.8)
                        }
                    }
                }
            })

        logger.info(f"Processing chat request: {message}")

        if not message and not project_details:
            return jsonify({
                'success': False,
                'error': 'Message or project details required'
            }), 400

        # Initial request for finding freelancers
        if 'find' in message.lower() or 'need' in message.lower():
            # Extract initial skills from message
            extracted_skills = model.skill_extractor.extract_skills(message)
            
            return jsonify({
                'success': True,
                'response': {
                    'type': 'project_details_request',
                    'text': "I'll help you find the perfect freelancer. Please provide the following details:",
                    'requiredInputs': {
                        'skills': {
                            'type': 'skill_list',
                            'message': "What skills are you looking for? (comma-separated)",
                            'initial': extracted_skills
                        },
                        'budget': {
                            'type': 'budget_range',
                            'message': "What's your budget range? (e.g., $30-100/hr)"
                        },
                        'complexity': {
                            'type': 'select',
                            'message': "How complex is your project?",
                            'options': ['low', 'medium', 'high']
                        },
                        'timeline': {
                            'type': 'number',
                            'message': "What's your project timeline in days?"
                        }
                    }
                }
            })

        # Handle submitted project details
        if project_details:
            # Create project from details
            project = Project(
                id=str(uuid.uuid4()),
                description=message,
                required_skills=project_details.get('skills', []),
                budget_range=parse_budget_range(project_details.get('budget', '')),
                complexity=project_details.get('complexity', 'medium'),
                timeline=int(project_details.get('timeline', 30))
            )

            # Find matches
            matches = await model.find_top_matches(project)
            
            if not matches:
                return jsonify({
                    'success': True,
                    'response': {
                        'type': 'text',
                        'text': "I couldn't find any matches for your criteria. Try adjusting your requirements."
                    }
                })

            return jsonify({
                'success': True,
                'response': {
                    'type': 'freelancerList',
                    'text': f"I found {len(matches)} matching freelancers:",
                    'freelancers': matches
                }
            })

        # Default response
        return jsonify({
            'success': True,
            'response': {
                'type': 'text',
                'text': "I can help you find freelancers. Just tell me what kind of project you need help with."
            }
        })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def analyze_intent(message: str) -> str:
    """Analyze user message intent"""
    intents = {
        'find_freelancers': ['find', 'need', 'looking', 'looking for' 'search', 'hire', 'Find me', 'find me'],
        'get_help': ['help', 'guide', 'explain'],
        'interview': ['interview', 'meet', 'talk'],
    }
    
    message_lower = message.lower()
    for intent, keywords in intents.items():
        if any(keyword in message_lower for keyword in keywords):
            return intent
    return 'general'

def format_user_matches(matches: List[Dict]) -> List[Dict]:
    """Enhanced user match formatting with more details"""
    return [{
        'id': match['user'].id,
        'name': match['user'].name,
        'jobTitle': match['user'].job_title,
        'skills': match['user'].skills,
        'hourlyRate': match['user'].hourly_rate,
        'rating': match['user'].rating,
        'score': float(match['combined_score']),
        'profileUrl': match['user'].profile_url,
        'availability': match['user'].availability,
        'totalJobs': match['user'].total_sales,
        'matchDetails': {
            'skillMatch': format_skill_match(match.get('skill_overlap', 0)),
            'experienceScore': match.get('experience_score', 0.0),
            'ratingScore': match.get('rating_score', 0.0),
            'matchPercentage': round(float(match['combined_score']) * 100, 1),
            'relevantExperience': format_experience_details(match['user']),
            'availability': format_availability(match['user'])
        }
    } for match in matches]

def generate_project_recommendations(project_details: Dict) -> List[str]:
    """
    Generate recommendations based on project analysis.
    
    Args:
        project_details (Dict): Project details including complexity, budget, etc.
        
    Returns:
        List[str]: List of recommendations
    """
    recommendations = []
    
    # Budget recommendations
    budget_min, budget_max = project_details['budget_range']
    if budget_max - budget_min < 100:
        recommendations.append("Consider widening your budget range to attract more qualified users")
    
    # Skills recommendations
    if len(project_details['required_skills']) > 7:
        recommendations.append("Consider breaking down the project into smaller phases or multiple specialists")
    
    # Complexity-based recommendations
    if project_details['complexity'] == 'high':
        recommendations.append("Recommend seeking users with 5+ years of experience")
        recommendations.append("Consider implementing milestone-based payments")
    
    return recommendations

@app.errorhandler(Exception)
def handle_error(error: Exception) -> tuple:
    """
    Global error handler for all endpoints.
    
    Args:
        error (Exception): The caught exception
        
    Returns:
        tuple: Error response and status code
    """
    logger.error(f"Unhandled error: {str(error)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': str(error)
    }), 500

# Add the missing functions
def format_skill_match(overlap_count: int) -> Dict[str, Any]:
    """Format skill match details"""
    if not isinstance(overlap_count, (int, float)):
        overlap_count = 0
    return {
        'count': overlap_count,
        'percentage': min(100, (overlap_count * 100) / 5),  # Normalize to percentage
        'level': 'high' if overlap_count >= 4 else 'medium' if overlap_count >= 2 else 'low'
    }

def generate_alternative_suggestions(project_details: Dict) -> List[str]:
    """Generate alternative search suggestions based on project details"""
    suggestions = []
    
    # Suggest broader skill combinations
    if len(project_details['required_skills']) > 3:
        core_skills = project_details['required_skills'][:3]
        suggestions.append(f"Find user with {', '.join(core_skills)}")
    
    # Suggest based on budget
    budget_min, budget_max = project_details['budget_range']
    if budget_max < 500:
        suggestions.append(f"Find user with budget up to ${budget_max * 1.5}")
    
    # Suggest based on complexity
    if project_details['complexity'] == 'high':
        suggestions.append("Find experienced user for complex project")
    
    # Add a general suggestion
    suggestions.append("Show all available users")
    
    return suggestions

def format_experience_details(user) -> Dict[str, Any]:
    """Format user's experience details"""
    return {
        'yearsOfExperience': user.experience,
        'totalProjects': user.total_sales,
        'successRate': user.rating,
        'relevantSkills': user.skills
    }

def format_availability(user) -> Dict[str, Any]:
    """Format user's availability status"""
    return {
        'status': 'Available' if user.availability else 'Busy',
        'immediateStart': user.availability,
        'hourlyRate': user.hourly_rate
    }

def parse_budget_range(budget_str: str) -> tuple:
    """Parse budget range from string input"""
    try:
        # Remove $ and /hr, split on hyphen or dash
        parts = budget_str.replace('$', '').replace('/hr', '').split('-')
        if len(parts) == 2:
            return (float(parts[0].strip()), float(parts[1].strip()))
        elif len(parts) == 1:
            value = float(parts[0].strip())
            return (value, value * 1.5)
        return (30, 100)  # reasonable default range
    except:
        return (30, 100)  # reasonable default range

# Modify the initialization part
if __name__ == '__main__':
    try:
        # Create and set event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Initialize the model
        loop.run_until_complete(initialize_model())
        
        # Start the Flask app
        app.run(
            host=Config.FLASK_HOST,
            port=Config.FLASK_PORT,
            debug=Config.DEBUG,
            use_reloader=False  # Prevent duplicate initialization
        )
    except Exception as e:
        logger.error(f"Startup error: {e}")
        sys.exit(1)
    finally:
        loop.close()