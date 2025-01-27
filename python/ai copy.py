from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import asyncio
import logging
from config import Config
from upworkModel import UpworkIntegrationModel
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
            logger.warning("API_KEY is not configured, using default key")
            Config.API_KEY = 'dev_key_123'
            
        model = UpworkIntegrationModel(
            api_url=Config.API_URL,
            api_key=Config.API_KEY
        )
        
        await model.initialize()
            
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}", exc_info=True)
        raise

@app.route('/ai/chat', methods=['POST'])
@async_route
async def chat():
    try:
        data = request.json
        message = data.get('message', '').strip()
        project_details = data.get('projectDetails')
        page = int(data.get('page', 1))

        if not message and not project_details:
            return jsonify({
                'success': False,
                'error': 'Please provide a message or project details'
            }), 400

        response = await model.handle_chat_request(message, project_details, page)
        
        # Ensure response includes required form fields
        if response.get('success') and 'response' in response:
            if not project_details:  # Only for initial messages
                response['response'].update({
                    'showProjectForm': True,
                    'type': 'job_analysis',
                    'autoSelectSkills': True
                })
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': "Sorry, I encountered an error. Please try again."
        }), 500

@app.route('/ai/analyze-job', methods=['POST'])
@async_route
async def analyze_job():
    try:
        data = request.json
        job_title = data.get('job_title', '').strip()

        if not job_title:
            return jsonify({
                'success': False,
                'error': 'Job title is required'
            }), 400

        if not model:
            await initialize_model()

        analysis = await model.analyze_job(job_title)
        
        # Keep all fields and add autoSelectSkills flag
        formatted_response = {
            'success': True,
            'response': {
                **analysis,  # Preserve all analysis fields
                'autoSelectSkills': True,  # Flag to auto-select skills
                'type': 'job_analysis',
                'text': 'Based on your requirements, here are the details:'
            }
        }

        return jsonify(formatted_response)

    except Exception as e:
        logger.error(f"Job analysis error: {e}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred while analyzing your request.'
        }), 500

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': str(error)
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
