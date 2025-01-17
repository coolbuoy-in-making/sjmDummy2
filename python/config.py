import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_URL = os.getenv('BACKEND_API_URL', 'http://localhost:5000')
    API_KEY = os.getenv('API_KEY')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 8000))
    
    @staticmethod
    def initialize_app(app):
        app.config['CORS_ORIGINS'] = os.getenv('CORS_ORIGINS', 'http://localhost:5173').split(',')
    
    @staticmethod
    def validate():
        required = ['API_KEY']
        missing = [key for key in required if not getattr(Config, key)]
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")