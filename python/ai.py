from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
sys.path.append('../python')
from upworkModel import UpworkAI

app = Flask(__name__)
CORS(app)

# Initialize the AI model
ai_model = UpworkAI()

@app.route('/ai/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message')
        
        # Get response from your AI model
        response = ai_model.get_response(message)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=5000)