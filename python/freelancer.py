import socket
import sys
import json
import time
from typing import List
from sjm import SkillsExtract

# Function to generate professional questions using AI services
def generate_professional_questions(context: dict) -> List[str]:
    print(f"Generating questions for project: {context['project_description']}...")
    questions = []

    try:
        # Attempt to use Claude first
        print("Using Claude for question generation...")
        claude_questions = SkillsExtract.generate_ai_interview_questions(
            context['project_description'], context['freelancer_skills']
        )
        if claude_questions:
            questions = claude_questions
    except Exception as e:
        print(f"Claude failed: {e}")

    if not questions:
        try:
            # Fallback to ChatGPT
            print("Using ChatGPT as fallback for question generation...")
            gpt_questions = SkillsExtract.generate_ai_interview_questions(
                context['project_description'], context['freelancer_skills']
            )
            if gpt_questions:
                questions = gpt_questions
        except Exception as e:
            print(f"ChatGPT failed: {e}")

    # Final fallback to hardcoded questions
    if not questions:
        print("Using fallback hardcoded questions...")
        questions = [
            f"Can you describe your experience with {context['project_description']}?",
            f"How do your skills in {', '.join(context['freelancer_skills'])} apply to this project?",
            "How do you typically manage project deadlines?",
            "What is your preferred communication method for project updates?",
            "Can you provide an example of a similar project you successfully completed?"
        ]
        
    if context.get('client_message'):
        questions.append({context['client_message']})    


    print("Questions generated successfully.")
    return questions


def main(host, port):
    try:
        # Initialize socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(500)  # Set timeout
        for attempt in range(3):
            try:
                client_socket.connect((host, int(port)))
                print(f"Connected to server at {host}:{port}")
                print(f"You have an interview notification, please standby...")
                break
            except socket.error as e:
                print(f"Attempt {attempt + 1} to connect failed: {e}")
                time.sleep(2)
        else:
            print("Failed to connect after 3 attempts.")
            return

        # Receive and validate context
        context_json = client_socket.recv(4096).decode('utf-8')
        if not context_json:
            print("No data received from server.")
            return
        
        try:
            context = json.loads(context_json)
        except json.JSONDecodeError:
            print("Invalid JSON received.")
            return

        # Process context and questions
        print("\n--- Interview Preparation ---")
        try:
            print(f"Project: {context['project_description']}")
            print(f"Freelancer: {context['freelancer_username']} ({context['freelancer_job_title']})")
        except KeyError as e:
            print(f"Missing expected field in context: {e}")
            return

        questions = generate_professional_questions(context)
        client_socket.sendall(json.dumps(questions).encode('utf-8'))

        # Handle answers and results
        answers = {q: input(f"Answer to '{q}': ") for q in questions}
        client_socket.sendall(json.dumps(answers).encode('utf-8'))

        results_json = b""
        while True:
            part = client_socket.recv(4096)
            if not part:
                break
            results_json += part
        results = json.loads(results_json.decode('utf-8'))

        print(f"Score: {results.get('score', 'N/A')}")
        print("Details:", results)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python freelancer.py <host> <port>")
    else:
        main(sys.argv[1], sys.argv[2])
