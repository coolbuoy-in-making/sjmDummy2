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

# Add these imports at the top if not already present
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import nltk
from sjm import SkillsExtract

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
        await model.initialize_matching_engine()
        
        freelancers = model.freelancers
        logger.info(f"\n{'='*50}\nLoaded {len(freelancers)} freelancers:")
        for i, f in enumerate(freelancers[:10], 1):
            logger.info(f"\n{i}. {f.name} ({f.job_title})")
            logger.info(f"   Skills: {', '.join(f.skills)}")
            logger.info(f"   Rate: ${f.hourly_rate}/hr, Rating: {f.rating}%")
        if len(freelancers) > 10:
            logger.info(f"\n... and {len(freelancers)-10} more freelancers")
        logger.info(f"{'='*50}\n")
        
        logger.info("Model initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}", exc_info=True)
        raise

def extract_project_details(message: str) -> Dict:
    """Extract project details from message using regex and NLP"""
    project_id = str(uuid.uuid4())
    
    budget_pattern = r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:[-–]\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?))?' 
    budget_match = re.search(budget_pattern, message)
    
    if budget_match:
        budget_min = float(budget_match.group(1).replace(',', ''))
        if budget_match.group(2):
            budget_max = float(budget_match.group(2).replace(',', ''))
        else:
            budget_max = budget_min * 1.5
    else:
        budget_min = 15
        budget_max = 150

    skills = model.skill_extractor.extract_skills(message)
    
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
        "id": project_id,
        "description": message,
        "required_skills": expanded_skills,
        "budget_range": (budget_min, budget_max),
        "complexity": determine_project_complexity(message, expanded_skills),
        "timeline": extract_timeline(message)
    }

def determine_project_complexity(message: str, skills: List[str]) -> str:
    skill_count = len(skills)
    
    complex_indicators = ['complex', 'sophisticated', 'advanced', 'enterprise', 'scalable']
    simple_indicators = ['simple', 'basic', 'straightforward', 'easy']
    
    message_lower = message.lower()
    complex_count = sum(1 for indicator in complex_indicators if indicator in message_lower)
    simple_count = sum(1 for indicator in simple_indicators if indicator in message_lower)
    
    if skill_count >= 5 or complex_count >= 2:
        return 'high'
    elif skill_count <= 2 and simple_count >= 1:
        return 'low'
    else:
        return 'medium'

def extract_timeline(message: str) -> int:
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
            
    return 30

def extract_freelancer_id(message: str) -> Optional[str]:
    id_patterns = [
        r'freelancer[_-]?id[:\s]+([a-zA-Z0-9-]+)',
        r'interview\s+(?:freelancer\s+)?([a-zA-Z0-9-]+)',
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

    def update_context(self, freelancer_id: str, context_data: Dict):
        if freelancer_id not in self.context:
            self.context[freelancer_id] = {}
        self.context[freelancer_id].update(context_data)

    def get_context(self, freelancer_id: str) -> Dict:
        return self.context.get(freelancer_id, {})

conversation_context = ConversationContext()

JOB_RELATED_WORDS = {
    'need', 'needed', 'looking', 'seek', 'seeking', 'hire', 'hiring', 'want',
    'required', 'requires', 'requiring', 'searching', 'urgently', 'immediate',
    'position', 'role', 'job', 'project', 'work', 'opportunity', 'gig',
    'freelancer', 'developer', 'designer', 'expert', 'specialist', 'professional'
}

def detect_hiring_intent(message: str, skill_extractor: SkillsExtract) -> Dict:
    message_lower = message.lower()
    words = set(word_tokenize(message_lower))
    
    skills = skill_extractor.extract_skills(message)
    
    has_job_words = bool(words & JOB_RELATED_WORDS)
    
    has_skills = len(skills) > 0
    
    return {
        'has_hiring_intent': has_job_words or has_skills,
        'detected_skills': skills,
        'job_words': list(words & JOB_RELATED_WORDS)
    }
    
async def handle_project_confirmation(message: str, project_details: Dict, page: int = 1) -> Dict:
    """Handle project details confirmation with improved validation and logging"""
    try:
        if not project_details:
            return {
                'type': 'error',
                'text': "No project details found. Please provide project requirements first."
            }

        logger.info("\n" + "="*50)
        logger.info("Starting Freelancer Search")
        logger.info("="*50)

        # Validate and extract budget
        budget_str = project_details.get('budget', '')
        if isinstance(budget_str, (tuple, list)):
            budget_range = budget_str
        else:
            budget_range = parse_budget_range(str(budget_str))
        
        logger.info(f"\nBudget Range: ${budget_range[0]}-${budget_range[1]}/hr")

        # Clean skills data
        input_skills = project_details.get('skills', [])
        if isinstance(input_skills, str):
            # Remove brackets, quotes and × characters
            cleaned_skills = re.sub(r'[\[\]\"×]', '', input_skills)
            input_skills = [s.strip() for s in cleaned_skills.split(',')]
        elif isinstance(input_skills, list):
            input_skills = [re.sub(r'[\[\]\"×]', '', s).strip() for s in input_skills]
        else:
            input_skills = []

        logger.info(f"Cleaned Input Skills: {input_skills}")

        # Use NLP to validate and expand skills
        validated_skills = []
        expanded_skills = set()
        
        for skill in input_skills:
            skill = skill.strip()
            if not skill:  # Skip empty skills
                continue
                
            verification = model.skill_extractor.verify_keyword(skill)
            if verification['exists']:
                validated_skills.extend(verification['skills'])
                related = await model.get_related_skills([skill])
                expanded_skills.update(related)

        # Add expanded skills if we need more
        if len(validated_skills) < 3:
            validated_skills.extend(list(expanded_skills)[:3])

        logger.info(f"Validated Skills: {validated_skills}")
        logger.info(f"Expanded Skills: {list(expanded_skills)}")

        # Create Project object
        project = Project(
            id=str(uuid.uuid4()),
            description=f"Project requiring skills in: {', '.join(validated_skills)}",
            required_skills=validated_skills,
            budget_range=budget_range,
            complexity=project_details.get('complexity', 'medium'),
            timeline=int(project_details.get('timeline', 30))
        )

        logger.info("\nSearching for matching freelancers...")
        
        # Calculate pagination
        items_per_page = 5
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page

        # Find matches with detailed evaluation logging
        matches = []
        total_evaluated = 0
        
        for freelancer in model.freelancers:
            total_evaluated += 1
            logger.info(f"\nEvaluating Freelancer {total_evaluated}: {freelancer.name}")
            logger.info(f"Job Title: {freelancer.job_title}")
            logger.info(f"Skills: {', '.join(freelancer.skills)}")
            logger.info(f"Rate: ${freelancer.hourly_rate}/hr")
            
            # Budget check
            if not (budget_range[0] <= freelancer.hourly_rate <= budget_range[1]):
                logger.info("[X] Rejected: Rate outside budget range")  # Instead of ❌
                continue
            logger.info("[✓] Budget: Within range")  # Instead of ✓
            
            # Skill matching
            matched_skills = set(validated_skills) & set(s.lower() for s in freelancer.skills)
            skill_match_score = len(matched_skills) / len(validated_skills) if validated_skills else 0
            
            logger.info(f"Matched Skills: {matched_skills}")
            logger.info(f"Skill Match Score: {skill_match_score:.2f}")
            
            # Experience and rating evaluation
            exp_score = min(freelancer.experience / 10.0, 1.0)
            rating_score = freelancer.rating / 5.0
            
            logger.info(f"Experience Score: {exp_score:.2f}")
            logger.info(f"Rating Score: {rating_score:.2f}")
            
            # Calculate combined score
            combined_score = (
                skill_match_score * 0.5 +
                exp_score * 0.25 +
                rating_score * 0.25
            )
            
            logger.info(f"Combined Score: {combined_score:.2f}")
            
            # Add if score is good enough
            if combined_score > 0.3:  # Adjust threshold as needed
                matches.append({
                    'id': str(freelancer.id),  # Add ID
                    'name': freelancer.name,
                    'jobTitle': freelancer.job_title,
                    'skills': freelancer.skills,
                    'hourlyRate': float(freelancer.hourly_rate),
                    'rating': float(freelancer.rating),
                    'experience': int(freelancer.experience),  # Ensure integer
                    'availability': bool(freelancer.availability),
                    'totalJobs': int(freelancer.total_sales),  # Add total jobs
                    'profileUrl': freelancer.profile_url,  # Add profile URL
                    'desc': freelancer.desc,  # Add description
                    'matchDetails': {
                        'matchPercentage': round(combined_score * 100, 1),
                        'skillMatch': {
                            'matched': list(matched_skills),
                            'suggested': list(set(freelancer.skills) - set(validated_skills))[:5],
                            'count': len(matched_skills)
                        },
                        'experienceScore': exp_score,
                        'ratingScore': rating_score,
                        'contentScore': skill_match_score,
                        'collaborativeScore': (exp_score + rating_score) / 2
                    }
                })
                logger.info("[✓] Added to matches")  # Instead of ✓
            else:
                logger.info("[X] Rejected: Score too low")  # Instead of ❌

        logger.info("\n" + "="*50)
        logger.info(f"Search Complete: Found {len(matches)} matches from {total_evaluated} freelancers")
        logger.info("="*50 + "\n")

        if not matches:
            # Get alternative suggestions with expanded skills
            expanded_matches = await model.get_related_skills(validated_skills)
            
            return {
                'type': 'no_matches',
                'text': "I couldn't find exact matches. Consider these alternatives:",
                'suggestions': generate_alternative_suggestions({
                    'required_skills': validated_skills,
                    'budget_range': budget_range,
                    'complexity': project_details.get('complexity', 'medium')
                }),
                'related_skills': expanded_matches,
                'adjustments': [
                    f"Current budget range (${budget_range[0]}-${budget_range[1]}/hr) might be limiting",
                    "Consider these related skills: " + ", ".join(list(expanded_skills)[:5]),
                    "Try broadening your search criteria"
                ]
            }

        # Sort matches by score and paginate
        matches.sort(key=lambda x: x['matchDetails']['matchPercentage'], reverse=True)
        paginated_matches = matches[start_idx:end_idx]
        total_pages = (len(matches) + items_per_page - 1) // items_per_page

        # Get related skills suggestions
        suggested_skills = set()
        for match in matches[:3]:  # Use top 3 matches for suggestions
            suggested_skills.update(match.get('skills', []))
        suggested_skills = suggested_skills - set(validated_skills)

        return {
            'type': 'freelancerList',
            'text': f"Found {len(matches)} freelancers matching your requirements:",
            'freelancers': paginated_matches,
            'pagination': {
                'currentPage': page,
                'totalPages': total_pages,
                'totalMatches': len(matches),
                'hasMore': page < total_pages
            },
            'suggestions': {
                'skills': list(suggested_skills)[:5],
                'actions': generate_alternative_suggestions(project_details)
            }
        }

    except Exception as e:
        logger.error(f"Error handling project confirmation: {e}")
        logger.error(f"Project details: {project_details}")
        return {
            'type': 'error',
            'text': "An error occurred while processing your request. Please try again."
        }

def analyze_intent(message: str) -> str:
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

def format_freelancer_matches(matches: List[Dict]) -> List[Dict]:
    return [{
        'id': match['freelancer'].id,
        'name': match['freelancer'].name,
        'jobTitle': match['freelancer'].job_title,
        'skills': match['freelancer'].skills,
        'hourlyRate': match['freelancer'].hourly_rate,
        'rating': match['freelancer'].rating,
        'score': float(match['combined_score']),
        'profileUrl': match['freelancer'].profile_url,
        'availability': match['freelancer'].availability,
        'totalJobs': match['freelancer'].total_sales,
        'matchDetails': {
            'skillMatch': format_skill_match(match.get('skill_overlap', 0)),
            'experienceScore': match.get('experience_score', 0.0),
            'ratingScore': match.get('rating_score', 0.0),
            'matchPercentage': round(float(match['combined_score']) * 100, 1),
            'relevantExperience': format_experience_details(match['freelancer']),
            'availability': format_availability(match['freelancer'])
        }
    } for match in matches]

def generate_project_recommendations(project_details: Dict) -> List[str]:
    recommendations = []
    
    budget_min, budget_max = project_details['budget_range']
    if budget_max - budget_min < 100:
        recommendations.append("Consider widening your budget range to attract more qualified freelancers")
    
    if len(project_details['required_skills']) > 7:
        recommendations.append("Consider breaking down the project into smaller phases or multiple specialists")
    
    if project_details['complexity'] == 'high':
        recommendations.append("Recommend seeking freelancers with 5+ years of experience")
        recommendations.append("Consider implementing milestone-based payments")
    
    return recommendations

@app.errorhandler(Exception)
def handle_error(error: Exception) -> tuple:
    logger.error(f"Unhandled error: {str(error)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': str(error)
    }), 500

def format_skill_match(overlap_count: int) -> Dict[str, Any]:
    if not isinstance(overlap_count, (int, float)):
        overlap_count = 0
    return {
        'count': overlap_count,
        'percentage': min(100, (overlap_count * 100) / 5),
        'level': 'high' if overlap_count >= 4 else 'medium' if overlap_count >= 2 else 'low'
    }

def generate_alternative_suggestions(project_details: Dict) -> List[str]:
    suggestions = []
    
    if len(project_details['required_skills']) > 3:
        core_skills = project_details['required_skills'][:3]
        suggestions.append(f"Find freelancer with {', '.join(core_skills)}")
    
    budget_min, budget_max = project_details['budget_range']
    if budget_max < 500:
        suggestions.append(f"Find freelancer with budget up to ${budget_max * 1.5}")
    
    if project_details['complexity'] == 'high':
        suggestions.append("Find experienced freelancer for complex project")
    
    suggestions.append("Show all available freelancers")
    
    return suggestions

def format_experience_details(freelancer) -> Dict[str, Any]:
    return {
        'yearsOfExperience': freelancer.experience,
        'totalProjects': freelancer.total_sales,
        'successRate': freelancer.rating,
        'relevantSkills': freelancer.skills
    }

def format_availability(freelancer) -> Dict[str, Any]:
    return {
        'status': 'Available' if freelancer.availability else 'Busy',
        'immediateStart': freelancer.availability,
        'hourlyRate': freelancer.hourly_rate
    }

def parse_budget_range(budget_str: str) -> tuple:
    try:
        parts = budget_str.replace('$', '').replace('/hr', '').split('-')
        if len(parts) == 2:
            return (float(parts[0].strip()), float(parts[1].strip()))
        elif len(parts) == 1:
            value = float(parts[0].strip())
            return (value, value * 1.5)
        return (30, 100)
    except:
        return (30, 100)

def clean_skills_list(skills):
    """Clean a list of skills, removing unwanted characters"""
    if not skills:
        return []
    if isinstance(skills, str):
        skills = [skills]
    return [
        re.sub(r'[\[\]"×\+]', '', skill).strip()
        for skill in skills
        if skill and isinstance(skill, str)
    ]

@app.route('/ai/chat', methods=['POST'])
@async_route
async def chat():
    try:
        data = request.json
        message = data.get('message', '').strip()  # Keep original case
        project_details = data.get('projectDetails')
        page = int(data.get('page', 1))  # Add page parameter

        # If requesting next page of results
        if message.lower() == 'load_more' and project_details:
            next_page = page + 1
            # Ensure matches includes pagination info
            matches = model.matching_engine.get_next_matches(page=next_page)
            
            if matches['matches']:
                return jsonify({
                    'success': True,
                    'response': {
                        'type': 'freelancerList',
                        'text': f"Here are more matching freelancers (page {next_page}):",
                        'freelancers': matches['matches'],
                        'pagination': {
                            'currentPage': next_page,
                            'totalPages': matches['pagination']['total_pages'],
                            'hasMore': matches['pagination']['has_next']
                        },
                        'suggestions': {
                            'skills': model.skill_extractor.get_suggested_skills(project_details),
                            'actions': generate_alternative_suggestions(project_details)
                        }
                    }
                })

        # Handle project details confirmation with pagination
        if message.lower() == 'confirm project details' and project_details:
            response = await handle_project_confirmation(message, project_details, page)
            # Ensure pagination info is included
            if response.get('type') == 'freelancerList':
                response['pagination'] = {
                    'currentPage': page,
                    'totalPages': response.get('totalPages', 1),
                    'hasMore': page < response.get('totalPages', 1)
                }
            return jsonify({
                'success': True,
                'response': response
            })

        logger.info(f"\nProcessing chat request:")
        logger.info(f"Message: {message}")

        # Handle project details confirmation
        if message.lower() == 'confirm project details' and project_details:
            response = await handle_project_confirmation(message, project_details)
            return jsonify({
                'success': True,
                'response': response
            })

        if message.lower() in model.greetings:
            return jsonify({
                'success': True,
                'response': {
                    'type': 'text',
                    'text': "Hi! I can help you find freelancers. What kind of skills or expertise are you looking for?"
                }
            })

        try:
            # Analyze message
            analysis = await model.analyze_job_title(message)
            logger.debug(f"Analysis result: {analysis}")

            if analysis['type'] == 'error':
                return jsonify({
                    'success': False,
                    'error': analysis['message']
                }), 500

            if analysis['type'] == 'no_match':
                return jsonify({
                    'success': True,
                    'response': {
                        'type': 'suggestions',
                        'text': analysis['message'],
                        'suggestions': analysis['suggestions'],
                        'popular_categories': analysis['popular_categories'],
                        'suggestedSkills': analysis.get('suggested_skills', [])
                    }
                })

            # Success case with matches
            if analysis['type'] == 'success':
                return jsonify({
                    'success': True,
                    'response': {
                        'type': 'job_analysis',
                        'text': f"Based on '{message}':",
                        'data': {
                            'detectedSkills': analysis['suggested_skills'],
                            'suggestedSkills': analysis['suggested_skills'],
                            'hourlyRange': analysis['avg_hourly_range'],
                            'sampleProfiles': analysis['sample_profiles'],
                            'skillRequirements': analysis['skill_requirements'],
                            'totalMatches': analysis['total_matches']
                        }
                    }
                })

            # Fallback response
            return jsonify({
                'success': True,
                'response': {
                    'type': 'text',
                    'text': "I'm not sure how to help with that. Try asking for specific skills or job titles like 'React developer' or 'UI designer'."
                }
            })

        except Exception as e:
            logger.error(f"Error analyzing request: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': "Sorry, I encountered an error. Please try again."
        }), 500

if __name__ == '__main__':
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        loop.run_until_complete(initialize_model())
        
        app.run(
            host=Config.FLASK_HOST,
            port=Config.FLASK_PORT,
            debug=Config.DEBUG,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Startup error: {e}")
        sys.exit(1)
    finally:
        loop.close()

@app.route('/ai/analyze-job', methods=['POST'])
@async_route
async def analyze_job():
    try:
        data = request.json
        job_title = data.get('jobTitle', '').strip()

        if not job_title:
            return jsonify({
                'success': False,
                'error': 'Job title is required'
            }), 400

        # Analyze the job title
        analysis = await model.analyze_job_title(job_title)
        
        # Handle different response types
        if analysis['type'] == 'error':
            return jsonify({
                'success': False,
                'error': analysis['message']
            }), 500
            
        if analysis['type'] == 'greeting':
            return jsonify({
                'success': True,
                'response': {
                    'type': 'text',
                    'text': analysis['message']
                }
            })
            
        if analysis['type'] == 'no_match':
            return jsonify({
                'success': True,
                'response': {
                    'type': 'suggestions',
                    'text': analysis['message'],
                    'suggestions': analysis['suggestions'],
                    'popular_categories': analysis['popular_categories'],
                    'suggestedSkills': analysis.get('suggested_skills', [])
                }
            })

        # Success case
        return jsonify({
            'success': True,
            'response': {
                'type': 'job_analysis',
                'text': f"Based on '{job_title}'. Here are some details to help refine your search:",
                'data': {
                    'suggestedSkills': analysis['suggested_skills'],
                    'hourlyRange': analysis['avg_hourly_range'],
                    'sampleProfiles': analysis['sample_profiles'],
                    'skillRequirements': analysis.get('skill_requirements', []),
                    'totalMatches': analysis['total_matches']
                }
            }
        })

    except Exception as e:
        logger.error(f"Job analysis error: {e}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred while analyzing your request. Please try again.'
        }), 500


# ... rest of existing code ...

def extract_skills(self, text: str) -> List[str]:
    """Enhanced word-by-word skill extraction"""
    matched_skills = set()
    words = word_tokenize(text.lower())
    
    # Clean incoming text by removing brackets and special characters
    text = re.sub(r'[\[\]\"×]', '', text)
    
    # Create word combinations for checking
    combinations = []
    for i in range(len(words)):
        if words[i] not in self.stop_words:
            # Clean each word
            clean_word = words[i].strip('[]"×')
            combinations.append(clean_word)
            
            # Two-word combinations
            if i < len(words) - 1:
                clean_next = words[i+1].strip('[]"×')
                combinations.append(f"{clean_word} {clean_next}")
            
            # Three-word combinations
            if i < len(words) - 2:
                clean_next = words[i+1].strip('[]"×')
                clean_next2 = words[i+2].strip('[]"×')
                combinations.append(f"{clean_word} {clean_next} {clean_next2}")

    # Check each combination against known skills and job titles
    for combo in combinations:
        if combo in self.known_skills:
            matched_skills.add(combo)
        if combo in self.job_titles:
            matched_skills.add(combo)

    return list(matched_skills)