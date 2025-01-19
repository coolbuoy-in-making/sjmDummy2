import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Add DEBUG configuration
    DEBUG = True  # Force debug mode for development
    
    # Backend API configuration
    API_URL = os.getenv('BACKEND_API_URL', 'http://localhost:5000')
    API_KEY = os.getenv('API_KEY')
    
    # Flask configuration
    FLASK_PORT = int(os.getenv('FLASK_PORT', 8000))
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    
    # CORS configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5173').split(',')
    
    # AI Model configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
    
    @staticmethod
    def validate():
        """Validate configuration settings"""
        missing = []
        
        # Check required configurations
        if not Config.API_URL:
            Config.API_URL = 'http://localhost:5000'  # Default value
            
        if not Config.API_KEY:
            Config.API_KEY = 'dev_key_123'  # Set default dev key
            
        # Validate API URL format
        if not Config.API_URL.startswith(('http://', 'https://')):
            Config.API_URL = f'http://{Config.API_URL}'

    @staticmethod
    def get_cors_config():
        return {
            'origins': Config.CORS_ORIGINS,
            'methods': ['GET', 'POST', 'OPTIONS'],
            'allow_headers': ['Content-Type', 'Authorization'],
            'supports_credentials': True
        }