# testing environment
import os
import sys
import socket
import subprocess
from typing import Dict, List, Optional, Any
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

        # Initialize stop words (excluding important words)
        stop_words = set(stopwords.words('english'))
        self.stop_words = stop_words - {
            'need', 'needed', 'want', 'looking', 'developer', 'designer', 
            'manager', 'expert', 'senior', 'junior', 'level'
        }

        # Initialize other attributes
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        self.rake = Rake()
        self.manual_keywords = []
        self.job_titles = set()
        self.known_skills = set()

        logger.info("SkillsExtract initialized with custom stop words")

    def load_keywords_from_database(self, freelancers: List['freelancer']) -> None:
        """Load skills and job titles from actual database entries"""
        for f in freelancers:
            # Add job titles
            if f.job_title:
                self.job_titles.add(f.job_title.lower())
                # Also add individual words from job titles
                self.job_titles.update(word.lower() for word in f.job_title.split())
            
            # Add skills
            if f.skills:
                self.known_skills.update(skill.lower() for skill in f.skills)
        
        # Update manual keywords to include all known terms
        self.manual_keywords = list(self.known_skills | self.job_titles)
        
        logger.info(f"Loaded {len(self.job_titles)} job titles and {len(self.known_skills)} skills from database")

    def extract_skills(self, text: str) -> List[str]:
        """Enhanced word-by-word skill extraction"""
        matched_skills = set()
        words = word_tokenize(text.lower())
        
        # Create word combinations for checking
        combinations = []
        for i in range(len(words)):
            # Skip stopwords except for important terms
            if words[i] not in self.stop_words:
                # Single word
                combinations.append(words[i])
                
                # Two-word combinations
                if i < len(words) - 1:
                    combinations.append(f"{words[i]} {words[i+1]}")
                
                # Three-word combinations
                if i < len(words) - 2:
                    combinations.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        # Check each combination against known skills and job titles
        for combo in combinations:
            if combo in self.known_skills:
                matched_skills.add(combo)
            if combo in self.job_titles:
                matched_skills.add(combo)

        logger.debug(f"Extracted skills from '{text}': {matched_skills}")
        return list(matched_skills)

    def verify_keyword(self, keyword: str) -> Dict[str, Any]:
        """Enhanced keyword verification with word-by-word analysis"""
        words = word_tokenize(keyword.lower())
        matches = []
        
        # Check each word combination against skills and job titles
        for i in range(len(words)):
            if words[i] not in self.stop_words:
                # Single word
                current_word = words[i]
                if current_word in self.known_skills:
                    matches.append(('skill', current_word))
                if current_word in self.job_titles:
                    matches.append(('job_title', current_word))

                # Two-word combinations if possible
                if i < len(words) - 1:
                    two_words = f"{words[i]} {words[i+1]}"
                    if two_words in self.known_skills:
                        matches.append(('skill', two_words))
                    if two_words in self.job_titles:
                        matches.append(('job_title', two_words))
        
        if matches:
            # Group matches by type
            skill_matches = [m[1] for m in matches if m[0] == 'skill']
            job_matches = [m[1] for m in matches if m[0] == 'job_title']
            
            return {
                'exists': True,
                'matches': list(set(skill_matches + job_matches)),
                'skills': list(set(skill_matches)),
                'job_titles': list(set(job_matches)),
                'type': 'skill' if skill_matches else 'job_title'
            }
        
        # No matches found - find similar terms
        similar_terms = self._find_similar_terms(keyword)
        return {
            'exists': False,
            'similar_terms': similar_terms,
            'type': None,
            'matches': [],
            'skills': [],
            'job_titles': []
        }

    def _find_similar_terms(self, keyword: str) -> List[str]:
        """Find similar terms from known skills and job titles"""
        similar = []
        
        for term in self.manual_keywords:
            if (keyword in term or term in keyword or
                SequenceMatcher(None, keyword, term).ratio() > 0.8):
                similar.append(term)
        
        return similar[:5]  # Return top 5 similar terms

    def _find_related_skills(self, keyword: str) -> List[str]:
        """Find related skills based on co-occurrence in profiles"""
        # This would be implemented in the specific platform integration
        return []

    @classmethod
    def generate_ai_interview_questions(
        self, 
        project_description: str,
        freelancer_skills: List[str]
    ) -> List[str]:
        """Generate AI-powered interview questions"""
        questions = [
            f"How would you apply your {', '.join(freelancer_skills)} to this project?",
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
class freelancer:
    id: str
    freelancername: str
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
            'freelancername': self.freelancername,
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
        self.freelancer_tfidf = None

    def train(self, freelancer_data):
        all_texts = [freelancer.profile_text() for freelancer in freelancer_data]
        self.freelancer_tfidf = self.tfidf_vectorizer.fit_transform(all_texts)

    def predict(self, project_tfidf):
        similarities = cosine_similarity(project_tfidf, self.freelancer_tfidf).flatten()
        return similarities

class CollaborativeModel:
    def __init__(self):
        self.freelancer_data = None
        self.project_data = None
        self.interaction_matrix = None
        self.skill_similarity_matrix = None

    def train(self, project_data: List[Dict], freelancer_data: List[freelancer]):
        self.freelancer_data = freelancer_data
        self.project_data = project_data

        num_freelancers = len(freelancer_data)
        if num_freelancers == 0:
            logger.warning("No freelancers available for training.")
            self.interaction_matrix = np.zeros((num_freelancers, 2))
            return

        try:
            # Create skill similarity matrix
            all_skills = set()
            for f in freelancer_data:
                all_skills.update(set(f.skills))
            
            skill_matrix = np.zeros((num_freelancers, len(all_skills)))
            skill_to_idx = {skill: idx for idx, skill in enumerate(all_skills)}
            
            for i, f in enumerate(freelancer_data):
                for skill in f.skills:
                    if skill in skill_to_idx:
                        skill_matrix[i, skill_to_idx[skill]] = 1

            # Calculate skill similarity between freelancers
            self.skill_similarity_matrix = cosine_similarity(skill_matrix)

            # Combine with traditional metrics
            total_sales = np.array([f.total_sales for f in freelancer_data])
            ratings = np.array([f.rating for f in freelancer_data])

            # Normalize metrics
            total_sales_norm = self._normalize_array(total_sales)
            ratings_norm = ratings / 5.0  # Assuming rating is out of 5

            self.interaction_matrix = np.column_stack((
                total_sales_norm,
                ratings_norm,
                np.mean(self.skill_similarity_matrix, axis=1)
            ))

        except Exception as e:
            logger.error(f"Error training collaborative model: {e}")
            self.interaction_matrix = np.zeros((num_freelancers, 2))

    def _normalize_array(self, arr):
        if arr.max() > arr.min():
            return (arr - arr.min()) / (arr.max() - arr.min())
        return np.zeros_like(arr)

    def predict(self, project_description: str, project_skills: List[str]) -> List[float]:
        if self.interaction_matrix is None or self.interaction_matrix.size == 0:
            logger.warning("Interaction matrix is empty. Returning zero scores.")
            return [0.0] * len(self.freelancer_data)

        try:
            # Calculate skill match scores
            skill_scores = np.zeros(len(self.freelancer_data))
            for i, freelancer in enumerate(self.freelancer_data):
                matched_skills = set(project_skills) & set(freelancer.skills)
                skill_scores[i] = len(matched_skills) / max(len(project_skills), 1)

            # Combine with interaction matrix
            final_scores = (
                0.5 * skill_scores +  # 50% weight to skill matching
                0.3 * self.interaction_matrix[:, 0] +  # 30% to experience/sales
                0.2 * self.interaction_matrix[:, 1]    # 20% to ratings
            )

            return final_scores.tolist()

        except Exception as e:
            logger.error(f"Error in collaborative prediction: {e}")
            return [0.0] * len(self.freelancer_data)

class MatchingEngine:
    def __init__(self, freelancers: List[freelancer], projects: List[Project], skill_extractor: SkillsExtract, collaborative_model=None):
        """
        Initialize the matching engine with freelancers, projects, and skill extraction tools.

        Args:
            freelancers (List[freelancer]): List of freelancer objects.
            projects (List[Project]): List of project objects.
            skill_extractor (SkillsExtract): A skill extraction tool for analyzing project descriptions.
        """
        self.freelancers = freelancers
        self.projects = projects
        self.skill_extractor = skill_extractor

        # Models
        self.content_model = ContentBasedModel()
        self.collaborative_model = collaborative_model

        # Precompute TF-IDF vectors for freelancers
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            [freelancer.profile_text() for freelancer in freelancers]
        )
        self.current_matches = []  # Store all matches for pagination
        self.page_size = 5  # Number of freelancers per page

    @staticmethod
    def similar(a: str, b: str) -> float:
        """
        Compute a similarity score between two strings.
        """
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def refine_skill_matching(self, required_skills: List[str], freelancer_skills: List[str]) -> int:
        """
        Refine skill matching to account for partial matches.
        Returns the number of overlapping or similar skills.
        """
        overlap_count = sum(
            1 for req_skill in required_skills
            for freelancer_skill in freelancer_skills
            if self.similar(req_skill, freelancer_skill) > 0.7
        )
        return overlap_count

    def train_models(self):
        """
        Train both content-based and collaborative models.
        """
        # Train Content-Based Model
        self.content_model.train(self.freelancers)

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
        self.collaborative_model.train(simulated_project_data, self.freelancers)

    def match_freelancers(self, project: Project, weights: Dict[str, float] = None, job_title_matcher=None, page: int = 1) -> Dict[str, Any]:
        """
        Enhanced matching with pagination support
        Returns dict with matches and pagination info
        """
        try:
            weights = weights or {
                'skills': 0.45,
                'experience': 0.20,
                'rating': 0.15,
                'job_title': 0.10,
                'availability': 0.10
            }

            # Only compute matches if this is first page or matches aren't stored
            if page == 1 or not self.current_matches:
                all_matches = []
                project_skills = set(s.lower() for s in project.required_skills)

                for freelancer in self.freelancers:
                    # ...existing matching logic...
                    matched_skills = project_skills & set(s.lower() for s in freelancer.skills)
                    skill_match_score = len(matched_skills) / max(len(project_skills), 1)
                    
                    # Get scores
                    exp_score = min(freelancer.experience / 10.0, 1.0)
                    rating_score = freelancer.rating / 5.0
                    availability_score = 1.0 if freelancer.availability else 0.5
                    
                    # Calculate job title score if provided
                    job_title_score = 0.0
                    if job_title_matcher and skill_match_score < 0.5:
                        job_title_score = job_title_matcher(project.description, freelancer.job_title)

                    # Calculate weighted score
                    total_score = (
                        weights['skills'] * skill_match_score +
                        weights['experience'] * exp_score +
                        weights['rating'] * rating_score +
                        weights['job_title'] * job_title_score +
                        weights['availability'] * availability_score
                    )

                    # Store match details
                    match = {
                        'freelancer': freelancer,
                        'combined_score': total_score,
                        'skill_overlap': len(matched_skills),
                        'matched_skills': list(matched_skills),
                        'skill_score': skill_match_score,
                        'experience_score': exp_score,
                        'rating_score': rating_score,
                        'job_title_score': job_title_score,
                        'availability_score': availability_score
                    }
                    all_matches.append(match)

                # Sort matches by score
                all_matches.sort(key=lambda x: (x['combined_score'], x['skill_overlap']), reverse=True)
                self.current_matches = all_matches

            # Calculate pagination
            start_idx = (page - 1) * self.page_size
            end_idx = start_idx + self.page_size
            page_matches = self.current_matches[start_idx:end_idx]
            total_matches = len(self.current_matches)
            total_pages = (total_matches + self.page_size - 1) // self.page_size

            return {
                'matches': page_matches,
                'pagination': {
                    'current_page': page,
                    'total_pages': total_pages,
                    'total_matches': total_matches,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                }
            }

        except Exception as e:
            logger.error(f"Error in matching freelancers: {e}")
            return {
                'matches': [],
                'pagination': {
                    'current_page': 1,
                    'total_pages': 1,
                    'total_matches': 0,
                    'has_next': False,
                    'has_previous': False
                }
            }

    def get_next_matches(self, page: int = 1) -> Dict[str, Any]:
        """Get next page of matches from current results"""
        if not self.current_matches:
            return {
                'matches': [],
                'pagination': {
                    'current_page': 1,
                    'total_pages': 1,
                    'total_matches': 0,
                    'has_next': False,
                    'has_previous': False
                }
            }

        start_idx = (page - 1) * self.page_size
        end_idx = start_idx + self.page_size
        page_matches = self.current_matches[start_idx:end_idx]
        total_matches = len(self.current_matches)
        total_pages = (total_matches + self.page_size - 1) // self.page_size

        return {
            'matches': page_matches,
            'pagination': {
                'current_page': page,
                'total_pages': total_pages,
                'total_matches': total_matches,
                'has_next': page < total_pages,
                'has_previous': page > 1
            }
        }

    def get_top_matches(self, project: Project, top_n: int = 5) -> List[Dict]:
        """
        Get the top N freelancer matches for a project.

        Args:
            project (Project): The project for which to find matches.
            top_n (int, optional): Number of top matches to return. Defaults to 5.

        Returns:
            List[Dict]: A list of top N freelancers.
        """
        all_matches = self.match_freelancers(project)
        return all_matches[:top_n]


    def interview_and_evaluate(self, freelancer: freelancer, project: Project) -> Dict:
        """Evaluate freelancer suitability"""
        questions = self.skill_extractor.generate_ai_interview_questions(
            project.description,
            freelancer.skills
        )
        
        return {
            'freelancer': freelancer.dict(),
            'questions': questions,
            'skill_match': self.refine_skill_matching(
                project.required_skills,
                freelancer.skills
            )
        }
    
    def ask_professional_questions(self, freelancer: freelancer, project: Project) -> List[str]:
        questions = [
            "Can you describe your experience with this type of project?",
            "How do you handle tight deadlines in your work?",
            "What tools do you use for project management?"
        ]
        print("Questions: ", questions)
        return questions

    def collect_answers(self, questions: List[str]) -> Dict[str, str]:
        return {q: "freelancer's response to " + q for q in questions}

    def ask_for_portfolio(self) -> Optional[str]:
        return "Portfolio link or file submission URL"

    def evaluate_answers(self, answers: Dict[str, str]) -> float:
        score = sum(len(answer) for answer in answers.values()) / 100
        return round(score, 2)

    def ask_client_for_custom_questions(self) -> Optional[Dict[str, str]]:
        custom_questions = {
            "What is your preferred communication tool?": "freelancer's response"
        }
        return custom_questions

    def accept_or_reject_freelancer(self, freelancer: freelancer, project: Project):
        client_decision = input(f"Do you want to accept {freelancer.freelancername} for project {project.id}? (yes/no): ")
        if client_decision.lower() == 'yes':
            return True
        return False

    def hire_freelancer(self, freelancer: freelancer):
        print(f"Notification: {freelancer.freelancername} has been hired!")

def normalize_csv(file_path: str, csv_columns: Optional[Dict[str, str]] = None) -> List[freelancer]:
    import pandas as pd
    df = pd.read_csv(file_path)
    csv_columns = csv_columns or {
        'id': 'id',
        'freelancername': 'freelancername',
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

    freelancers = []
    for _, row in df.iterrows():
        try:
            freelancer = freelancer(
                id=row[csv_columns['id']],
                freelancername=row[csv_columns['freelancername']],
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
            freelancers.append(freelancer)
        except Exception as e:
            logger.warning(f"Skipping row due to error: {e}")
    return freelancers

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
            client_command = [sys.executable, "freelancer.py", self.host, str(self.port)]

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

