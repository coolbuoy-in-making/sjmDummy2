# testing environment
import os
import sys
import socket
import subprocess
from typing import Dict, List, Optional
import logging

# models

# similar model
from difflib import SequenceMatcher
from dataclasses import dataclass

import numpy as np

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rake_nltk import Rake


# Import AI services
import anthropic
from openai import OpenAI

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


# Ensure NLTK resources are downloaded
nltk.download(['punkt', 'stopwords'], quiet=True)

class SkillsExtract:
    def __init__(
        self,
        claude_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        # Load API keys securely
        self.claude_api_key = claude_api_key or os.getenv('CLAUDE_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

        # Ensure NLTK resources are downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        # Initialize vectorization tools
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.rake = Rake()

        # Manual keywords for initial skill extraction
        self.manual_keywords = [
            # Web Development
    "web development", "frontend development", "backend development", "full-stack development", "HTML", "CSS",
    "JavaScript", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "PHP", "Ruby on Rails", "ASP.NET",
    "Laravel", "WordPress", "Shopify", "eCommerce", "Web design", "UI/UX design", "Responsive design",

    # Programming Languages
    "Python", "JavaScript", "Java", "C++", "C#", "Ruby", "PHP", "Go", "Swift", "Kotlin", "R", "MATLAB", "Perl",
    "TypeScript", "Rust", "Scala", "Haskell", "Shell scripting", "Bash", "SQL", "NoSQL", "GraphQL",

    # Data Science & Machine Learning
    "data science", "machine learning", "deep learning", "artificial intelligence", "AI", "data analysis", "data engineering",
    "Big Data", "Hadoop", "Spark", "TensorFlow", "PyTorch", "scikit-learn", "Keras", "Pandas", "NumPy", "Matplotlib",
    "Seaborn", "data visualization", "statistics", "natural language processing", "NLP", "computer vision",

    # Cloud Computing
    "cloud computing", "AWS", "Amazon Web Services", "Google Cloud Platform", "GCP", "Microsoft Azure", "cloud architecture",
    "DevOps", "CI/CD", "Docker", "Kubernetes", "Terraform", "Ansible", "cloud security", "serverless architecture",

    # Cybersecurity
    "cybersecurity", "ethical hacking", "penetration testing", "network security", "cryptography", "incident response",
    "firewall management", "security operations", "SIEM", "SOC", "threat analysis", "vulnerability assessment",

    # Marketing & Content Creation
    "marketing", "digital marketing", "SEO", "search engine optimization", "content marketing", "social media marketing",
    "email marketing", "influencer marketing", "Google Ads", "Facebook Ads", "copywriting", "content writing", "video editing",
    "graphic design", "branding", "market research", "affiliate marketing",

    # Design
    "graphic design", "UI design", "UX design", "Figma", "Adobe Photoshop", "Adobe Illustrator", "Adobe XD", "Canva",
    "motion graphics", "3D design", "Blender", "AutoCAD", "Sketch", "prototyping", "interaction design",

    # Writing & Communication
    "writing", "content writing", "technical writing", "creative writing", "editing", "proofreading", "copywriting",
    "blogging", "academic writing", "speechwriting", "transcription", "translation", "grant writing", "business communication",

    # Project Management & Business
    "project management", "Agile", "Scrum", "Kanban", "Jira", "Trello", "Asana", "business analysis", "product management",
    "business strategy", "Lean", "Six Sigma", "change management", "operations management", "supply chain management",

    # Other Skills
    "game development", "Unity", "Unreal Engine", "VR", "AR", "IoT", "robotics", "blockchain", "smart contracts",
    "solidity", "data entry", "virtual assistant", "technical support", "customer support", "sales", "financial analysis",
    "stock trading", "investment analysis", "legal writing", "paralegal", "video production", "podcasting", "music production",
    "audio editing"
        ]

    def extract_skills(self, project_description: str) -> List[str]:
        """
        Advanced skill extraction with multiple strategies:
        1. Manual keyword matching
        2. NLTK RAKE keyword extraction
        3. NLTK-based text processing
        4. TF-IDF based extraction

        Args:
            project_description (str): Textual description of the project

        Returns:
            List[str]: Extracted and cleaned skills
        """
        # Normalize project description
        project_description = project_description.lower()

        # Step 1: Manual Keyword Matching
        manual_matched_skills = [
            keyword for keyword in self.manual_keywords
            if keyword.lower() in project_description
        ]

        # Step 2: NLTK RAKE Keyword Extraction
        self.rake.extract_keywords_from_text(project_description)
        rake_keywords = self.rake.get_ranked_phrases()

        # Step 3: NLTK-based text processing
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(project_description)
        nltk_keywords = [
            word for word in tokens
            if word.lower() not in stop_words
            and len(word) > 2
        ]

        # Step 4: TF-IDF Skill Extraction
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([project_description])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # Get top keywords by TF-IDF score
            top_indices = tfidf_scores.argsort()[-10:][::-1]
            tfidf_keywords = [feature_names[i] for i in top_indices]
        except Exception as e:
            tfidf_keywords = []

        # Combine all extraction methods
        all_skills = manual_matched_skills + rake_keywords + nltk_keywords + tfidf_keywords

        # Advanced filtering and cleaning
        filtered_skills = []
        for skill in all_skills:
            skill = skill.lower().strip()
            # Check if skill is in our manual keywords or derives from them
            if len(skill) > 2 and any(
                category.lower() in skill
                for category in self.manual_keywords
            ):
                filtered_skills.append(skill)

        # Remove duplicates and return
        return list(set(filtered_skills))

    @classmethod
    def generate_ai_interview_questions(
        self, 
        project_description: str,
        user_skills: List[str]
    ) -> List[str]:
        """Generate AI-powered interview questions"""
        questions = [
            f"How would you apply your {', '.join(user_skills)} to this project?",
            "What is your approach to project management and deadlines?",
            "How do you handle communication with clients?",
            "Can you describe similar projects you've completed?",
            "What would be your first steps if selected for this project?"
        ]
        
        # Add project-specific questions based on description
        project_questions = self._generate_project_specific_questions(project_description)
        questions.extend(project_questions)
        
        return questions


@dataclass
class user:
    id: str
    username: str
    name: str
    job_title: str
    skills: List[str]
    experience: int
    rating: float
    hourly_rate: float
    profile_url: str
    availability: bool
    total_sales: int 
    desc: str # Add default value

    def profile_text(self) -> str:
        """Return text representation for TF-IDF"""
        return f"{self.name} - {self.job_title}. {self.desc} Skills: {', '.join(self.skills)}"

    def dict(self) -> Dict[str, any]:
        return {
            'id': self.id,
            'username': self.username,
            'name': self.name,
            'job_title': self.job_title,
            'skills': self.skills,
            'experience': self.experience,
            'rating': self.rating,
            'hourly_rate': self.hourly_rate,
            'profile_url': self.profile_url,
            'availability': self.availability,
            'total_sales': self.total_sales,
        }


@dataclass
class Project:
    id: str
    description: str
    required_skills: List[str]
    budget_range: tuple
    complexity: str
    timeline: Optional[int] = None

class ContentBasedModel:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.user_tfidf = None

    def train(self, user_data):
        all_texts = [user.profile_text() for user in user_data]
        self.user_tfidf = self.tfidf_vectorizer.fit_transform(all_texts)

    def predict(self, project_tfidf):
        similarities = cosine_similarity(project_tfidf, self.user_tfidf).flatten()
        return similarities

class CollaborativeModel:
    def __init__(self):
        self.user_data = None
        self.project_data = None
        self.interaction_matrix = None

    def train(self, project_data: List[Dict], user_data: List[user]):
        self.user_data = user_data
        self.project_data = project_data

        num_users = len(user_data)
        num_projects = len(project_data)

        if num_projects == 0 or num_users == 0:
            logger.warning("No users or projects available for training.")
            self.interaction_matrix = np.zeros((num_users, 2))
            return

        try:
            total_sales = np.array([f.total_sales for f in user_data])
            ratings = np.array([f.rating for f in user_data])

            # Handle cases where min == max to avoid division by zero
            if total_sales.max() > total_sales.min():
                total_sales_norm = (total_sales - total_sales.min()) / (total_sales.max() - total_sales.min())
            else:
                total_sales_norm = np.zeros_like(total_sales)

            ratings_norm = ratings / 5.0  # Assuming rating is out of 5

            self.interaction_matrix = np.column_stack((total_sales_norm, ratings_norm))
        except Exception as e:
            logger.error(f"Error training collaborative model: {e}")
            self.interaction_matrix = np.zeros((num_users, 2))

    def predict(self, project_description: str, project_skills: List[str]) -> List[float]:
        """
        Predict match scores using collaborative filtering.
        """
        if self.interaction_matrix is None or self.interaction_matrix.size == 0:
            logger.warning("Interaction matrix is empty. Returning zero scores.")
            return [0.0] * len(self.user_data)

        # Compute average scores while handling potential NaN values
        scores = np.nanmean(self.interaction_matrix, axis=1)  # Avoid NaN propagation
        return np.nan_to_num(scores).tolist()

class MatchingEngine:
    def __init__(self, users: List[user], projects: List[Project], skill_extractor: SkillsExtract, collaborative_model=None):
        """
        Initialize the matching engine with users, projects, and skill extraction tools.

        Args:
            users (List[user]): List of user objects.
            projects (List[Project]): List of project objects.
            skill_extractor (SkillsExtract): A skill extraction tool for analyzing project descriptions.
        """
        self.users = users
        self.projects = projects
        self.skill_extractor = skill_extractor

        # Models
        self.content_model = ContentBasedModel()
        self.collaborative_model = collaborative_model

        # Precompute TF-IDF vectors for users
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            [user.profile_text() for user in users]
        )

    @staticmethod
    def similar(a: str, b: str) -> float:
        """
        Compute a similarity score between two strings.
        """
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def refine_skill_matching(self, required_skills: List[str], user_skills: List[str]) -> int:
        """
        Refine skill matching to account for partial matches.
        Returns the number of overlapping or similar skills.
        """
        overlap_count = sum(
            1 for req_skill in required_skills
            for user_skill in user_skills
            if self.similar(req_skill, user_skill) > 0.7
        )
        return overlap_count

    def train_models(self):
        """
        Train both content-based and collaborative models.
        """
        # Train Content-Based Model
        self.content_model.train(self.users)

        # Simulate historical project data for Collaborative Filtering
        simulated_project_data = [
            {
                "id": project.id,
                "description": project.description,
                "required_skills": project.required_skills,
                "budget_range": project.budget_range,
                "complexity": project.complexity,
            }
            for project in self.projects
        ]
        self.collaborative_model.train(simulated_project_data, self.users)

    def match_users(self, project: Project, weights: Dict[str, float] = None) -> List[Dict]:
        """
        Match users to a given project using a hybrid approach.
        """
        weights = weights or {'content': 0.3, 'collaborative': 0.4, 'experience': 0.2, 'rating': 0.1}
        weight_sum = sum(weights.values())
        if weight_sum > 1.0:
            logger.warning("Weights sum exceeds 1.0, normalizing weights.")
            weights = {k: v / weight_sum for k, v in weights.items()}

        # Extract required skills
        project_skills = self.skill_extractor.extract_skills(project.description)

        # Calculate TF-IDF vector for the project description
        project_tfidf = self.tfidf_vectorizer.transform([project.description])

        # Compute Content-Based Scores
        content_scores = self.content_model.predict(project_tfidf)

        # Compute Collaborative Scores
        collaborative_scores = self.collaborative_model.predict(project.description, project_skills)

        # Combine Scores with boosted weight for skill overlap
        final_scores = []
        for idx, user in enumerate(self.users):
            # Use refined skill matching logic
            skill_overlap_count = self.refine_skill_matching(project_skills, user.skills)
            skill_match_score = skill_overlap_count / len(project_skills) if project_skills else 0

            combined_score = (
                weights['content'] * content_scores[idx]
                + weights['collaborative'] * collaborative_scores[idx]
                + weights['experience'] * (user.experience / 10)
                + weights['rating'] * (user.rating / 5)
                + 0.2 * skill_match_score  # Boost for skill overlap
            )
            final_scores.append({
                'user': user,
                'combined_score': combined_score,
                'content_score': content_scores[idx],
                'collaborative_score': collaborative_scores[idx],
                'skill_overlap': skill_overlap_count,
            })

        # Sort and return top matches
        return sorted(final_scores, key=lambda x: x['combined_score'], reverse=True)

    def get_top_matches(self, project: Project, top_n: int = 5) -> List[Dict]:
        """
        Get the top N user matches for a project.

        Args:
            project (Project): The project for which to find matches.
            top_n (int, optional): Number of top matches to return. Defaults to 5.

        Returns:
            List[Dict]: A list of top N users.
        """
        all_matches = self.match_users(project)
        return all_matches[:top_n]


    def interview_and_evaluate(self, user: user, project: Project) -> Dict:
        """Evaluate user suitability"""
        questions = self.skill_extractor.generate_ai_interview_questions(
            project.description,
            user.skills
        )
        
        return {
            'user': user.dict(),
            'questions': questions,
            'skill_match': self.refine_skill_matching(
                project.required_skills,
                user.skills
            )
        }
    
    def ask_professional_questions(self, user: user, project: Project) -> List[str]:
        questions = [
            "Can you describe your experience with this type of project?",
            "How do you handle tight deadlines in your work?",
            "What tools do you use for project management?"
        ]
        print("Questions: ", questions)
        return questions

    def collect_answers(self, questions: List[str]) -> Dict[str, str]:
        return {q: "user's response to " + q for q in questions}

    def ask_for_portfolio(self) -> Optional[str]:
        return "Portfolio link or file submission URL"

    def evaluate_answers(self, answers: Dict[str, str]) -> float:
        score = sum(len(answer) for answer in answers.values()) / 100
        return round(score, 2)

    def ask_client_for_custom_questions(self) -> Optional[Dict[str, str]]:
        custom_questions = {
            "What is your preferred communication tool?": "user's response"
        }
        return custom_questions

    def accept_or_reject_user(self, user: user, project: Project):
        client_decision = input(f"Do you want to accept {user.username} for project {project.id}? (yes/no): ")
        if client_decision.lower() == 'yes':
            return True
        return False

    def hire_user(self, user: user):
        print(f"Notification: {user.username} has been hired!")

def normalize_csv(file_path: str, csv_columns: Optional[Dict[str, str]] = None) -> List[user]:
    import pandas as pd
    df = pd.read_csv(file_path)
    csv_columns = csv_columns or {
        'id': 'id',
        'username': 'username',
        'name': 'name',
        'job_title': 'job_title',
        'skills': 'skills',
        'experience': 'experience',
        'rating': 'rating',
        'hourly_rate': 'hourly_rate',
        'profile_url': 'profile_url',
        'availability': 'availability',
        'total_sales': 'total_sales'
    }

    users = []
    for _, row in df.iterrows():
        try:
            user = user(
                id=row[csv_columns['id']],
                username=row[csv_columns['username']],
                name=row[csv_columns['name']],
                job_title=row[csv_columns['job_title']],
                skills=row[csv_columns['skills']].split(','),
                experience=int(row.get(csv_columns['experience'], 0)),
                rating=float(row.get(csv_columns['rating'], 0)),
                hourly_rate=float(row.get(csv_columns['hourly_rate'], 0)),
                profile_url=row[csv_columns['profile_url']],
                availability=row[csv_columns['availability']],
                total_sales=int(row.get(csv_columns['total_sales'], 0))
            )
            users.append(user)
        except Exception as e:
            logger.warning(f"Skipping row due to error: {e}")
    return users

class Server:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.connection = None
        self.address = None
        self.is_connected = False

    def setup_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print("Server waiting for connection...")
            return True
        except Exception as e:
            print(f"Error setting up server: {e}")
            return False

    def start_server(self):
        """
        Start the server and wait for a client connection.
        Automatically launches a new terminal for the client process.
        """
        if self.setup_server():
            # Launch a new terminal for the client process
            self.start_client_in_new_terminal()

            try:
                self.connection, self.address = self.server_socket.accept()
                self.is_connected = True
                print(f"Connection from {self.address}")
                return self.connection
            except Exception as e:
                print(f"Error accepting connection: {e}")
                return None

    def start_client_in_new_terminal(self):
        try:
            client_command = [sys.executable, "user.py", self.host, str(self.port)]

            # Automatically open a new terminal with the argument
            if os.name == 'nt':  # Windows
                subprocess.Popen(["start", "cmd", "/k"] + client_command, shell=True)
            else:  # macOS/Linux
                subprocess.Popen(["gnome-terminal", "--"] + client_command)

            print("Client terminal started successfully.")
        except Exception as e:
            print(f"Error starting client in a new terminal: {e}")


    def send_message(self, message, is_server=True):
        if not self.is_connected:
            print("Not connected. Cannot send message.")
            return False

        try:
            socket_to_use = self.connection if is_server else self.client_socket
            if socket_to_use:
                socket_to_use.send(message.encode('utf-8'))
                return True
            else:
                print("No socket available to send message.")
                return False
        except Exception as e:
            print(f"Error sending message: {e}")
            return False

    def receive_message(self, is_server=True):
        if not self.is_connected:
            print("Not connected. Cannot receive message.")
            return None
        try:
            socket_to_use = self.connection if is_server else self.client_socket
            if socket_to_use:
                message = socket_to_use.recv(1024).decode('utf-8')
                return message
            else:
                print("No socket available to receive message.")
                return None
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None

    def close_connection(self):
        """
        Close server and client connections if they exist.
        """
        try:
            if self.connection:
                self.connection.close()
            if self.server_socket:
                self.server_socket.close()
            self.is_connected = False
        except Exception as e:
            print(f"Error closing connection: {e}")
