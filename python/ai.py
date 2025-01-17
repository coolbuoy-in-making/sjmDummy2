from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List, Optional
import json
import re
import sys
from config import Config
from upworkModel import UpworkIntegrationModel
from sjm import Project, SkillsExtract
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('fiverr_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize with API configuration
Config.validate()
model = UpworkIntegrationModel(
    api_url=Config.API_URL,
    api_key=Config.API_KEY
)

def extract_project_details(message: str) -> Dict:
    """Extract project details from message using regex and NLP"""
    # Basic extraction of budget using regex
    budget_pattern = r'\$(\d+)(?:[-\s]*(\d+))?' 
    budget_match = re.search(budget_pattern, message)
    
    budget_min = float(budget_match.group(1)) if budget_match else 0
    budget_max = float(budget_match.group(2)) if budget_match and budget_match.group(2) else budget_min * 2

    # Extract skills using SkillsExtract
    skills = skill_extractor.extract_skills(message)
    
    return {
        "description": message,
        "required_skills": skills,
        "budget_range": (budget_min, budget_max),
        "complexity": "medium"  # Default value
    }

@app.route('/ai/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').lower()
        user_type = data.get('userType', 'client')
        user_id = data.get('userId')

        # Handle different message intents
        if 'find' in message and ('freelancer' in message or 'developer' in message):
            project_details = extract_project_details(message)
            project = Project(
                id="temp_id",
                description=project_details["description"],
                required_skills=project_details["required_skills"],
                budget_range=project_details["budget_range"],
                complexity=project_details["complexity"]
            )
            matches = model.find_top_matches(project)
            
            # Format matches for frontend
            formatted_matches = [{
                'id': match['freelancer'].id,
                'name': match['freelancer'].name,
                'jobTitle': match['freelancer'].job_title,
                'skills': match['freelancer'].skills,
                'hourlyRate': match['freelancer'].hourly_rate,
                'rating': match['freelancer'].rating,
                'score': match['combined_score'],
                'profileUrl': match['freelancer'].profile_url
            } for match in matches]
            
            return jsonify({
                'success': True,
                'response': {
                    'type': 'freelancerList',
                    'freelancers': formatted_matches
                }
            })
            
        elif 'interview' in message:
            # Extract freelancer ID if provided in message
            freelancer_id = None  # Add logic to extract freelancer ID if needed
            questions = skill_extractor.generate_ai_interview_questions(
                message,
                []  # Add relevant skills here
            )
            return jsonify({
                'success': True,
                'response': {
                    'type': 'interview',
                    'questions': questions,
                    'freelancerId': freelancer_id
                }
            })
        else:
            suggestions = [
                "You can try:",
                "- 'Find a freelancer for...' to search for matching freelancers",
                "- 'Interview freelancer [name]' to start an interview",
                "- 'What are the best freelancers for [skill]?'"
            ]
            return jsonify({
                'success': True,
                'response': {
                    'type': 'text',
                    'text': "I can help you find freelancers or conduct interviews.\n" + "\n".join(suggestions)
                }
            })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(Exception)
def handle_error(error):
    return jsonify({
        'success': False,
        'error': str(error)
    }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)