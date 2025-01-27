from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import asyncio
import logging
import os
from datetime import datetime
from config import Config
from upworkModel import UpworkIntegrationModel
from functools import wraps
import nest_asyncio
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure asyncio for production/development
nest_asyncio.apply()

def create_app(config_object=None):
    """Factory function to create and configure the Flask app"""
    app = Flask(__name__)
    
    # Load configuration
    if config_object:
        app.config.from_object(config_object)
    else:
        # Default to production config
        app.config['PRODUCTION'] = os.getenv('FLASK_ENV', 'production') == 'production'
        app.config['DEBUG'] = not app.config['PRODUCTION']
        app.config['TESTING'] = False

    # Configure CORS
    CORS(app, resources={
        r"/*": {
            "origins": Config.CORS_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    # Configure for proxy servers (like Render's)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # Configure logging based on environment
    configure_logging(app.config['PRODUCTION'])

    return app

def configure_logging(is_production):
    """Configure logging based on environment"""
    log_level = logging.INFO if is_production else logging.DEBUG
    log_format = '%(asctime)s - %(levelname)s [%(name)s] %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if is_production:
        # In production, also log to file
        log_file = os.getenv('LOG_FILE', 'upwork_integration.log')
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

def async_route(f):
    """Decorator to handle async routes"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(f(*args, **kwargs))
    return wrapped

# Create the Flask app
app = create_app()
logger = logging.getLogger(__name__)

# Initialize model as None - will be initialized on first request
model = None

async def init_model():
    """Initialize the UpworkIntegrationModel"""
    global model
    if model is None:
        Config.validate()
        
        api_url = os.getenv('API_URL', Config.API_URL)
        api_key = os.getenv('API_KEY', Config.API_KEY)
        
        if not api_key and Config.DEBUG:
            api_key = 'dev_key_123'
            logger.warning("Using default development API key")
            
        model = UpworkIntegrationModel(
            api_url=api_url,
            api_key=api_key
        )
        await model.initialize()
        
        if not Config.DEBUG:
            logger.info("UpworkIntegrationModel initialized in production mode")
    return model

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'environment': 'production' if app.config['PRODUCTION'] else 'development',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/ai/chat', methods=['POST'])
@async_route
async def chat():
    """Chat endpoint with improved error handling"""
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

        if not model:
            await init_model()

        response = await model.handle_chat_request(
            message=message,
            project_details=project_details,
            page=page
        )
        
        if response.get('success') and 'response' in response:
            if not project_details:
                response['response'].update({
                    'showProjectForm': True,
                    'type': 'job_analysis',
                    'autoSelectSkills': True
                })
        
        return jsonify(response)

    except Exception as e:
        logger.exception("Error in chat endpoint")
        return jsonify({
            'success': False,
            'error': "An unexpected error occurred. Please try again."
        }), 500

@app.route('/ai/analyze-job', methods=['POST'])
@async_route
async def analyze_job():
    """Job analysis endpoint with improved error handling"""
    try:
        data = request.json
        job_title = data.get('job_title', '').strip()

        if not job_title:
            return jsonify({
                'success': False,
                'error': 'Job title is required'
            }), 400

        if not model:
            await init_model()

        analysis = await model.analyze_job(job_title)
        
        formatted_response = {
            'success': True,
            'response': {
                **analysis,
                'autoSelectSkills': True,
                'type': 'job_analysis',
                'text': 'Based on your requirements, here are the details:'
            }
        }

        return jsonify(formatted_response)

    except Exception as e:
        logger.exception("Error in analyze-job endpoint")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred while analyzing your request.'
        }), 500

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.exception("Unhandled error occurred")
    if app.config['PRODUCTION']:
        message = "An unexpected error occurred"
    else:
        message = str(error)
    
    return jsonify({
        'success': False,
        'error': message
    }), 500

def run_app():
    """Function to run the app with proper configuration"""
    host = Config.FLASK_HOST
    port = Config.FLASK_PORT
    debug = Config.DEBUG
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Startup error: {e}")
        sys.exit(1)
    finally:
        loop.close()

if __name__ == '__main__':
    run_app()