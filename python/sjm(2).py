# testing environment
import os
import sys
import socket
import subprocess
from typing import Dict, List, Optional, Any
import logging
import re

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

    def load_keywords_from_database(self, freelancers: List['Freelancer']) -> None:
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

    def clean_skill(self, skill: str) -> str:
        """Clean individual skill string"""
        # Remove brackets, quotes, × and other unwanted characters
        cleaned = re.sub(r'[\[\]"×\+\(\)]', '', skill.strip())
        
        # Special formatting for common tools/technologies
        tech_formats = {
            'adobe xd': 'Adobe XD',
            'blender': 'Blender',
            'figma': 'Figma',
            'color theory': 'Color Theory',
            'unreal engine': 'Unreal Engine',
            'react': 'React.js',
            'reactjs': 'React.js',
            'node': 'Node.js',
            'nodejs': 'Node.js',
            'vue': 'Vue.js',
            'vuejs': 'Vue.js',
            'typescript': 'TypeScript',
            'javascript': 'JavaScript',
            'nextjs': 'Next.js',
            'nuxtjs': 'Nuxt.js',
            'expressjs': 'Express.js',
            # ... existing mappings ...
        }
        
        cleaned_lower = cleaned.lower()
        if cleaned_lower in tech_formats:
            return tech_formats[cleaned_lower]
        
        # Handle multi-word skills
        words = cleaned.split()
        if len(words) > 1:
            return ' '.join(word.capitalize() for word in words)
        
        return cleaned.capitalize()

    def extract_skills(self, text: str) -> List[str]:
        """Enhanced skill extraction with proper formatting"""
        if not text:
            return []
            
        # Clean incoming text and split into words
        text = re.sub(r'[\[\]"×\+]', '', text)
        words = word_tokenize(text.lower())
        
        matched_skills = set()
        
        # Create word combinations
        combinations = []
        for i in range(len(words)):
            if words[i] not in self.stop_words:
                combinations.append(words[i])
                if i < len(words) - 1:
                    combinations.append(f"{words[i]} {words[i+1]}")
                if i < len(words) - 2:
                    combinations.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        # Match and format skills
        for combo in combinations:
            combo_lower = combo.lower()
            if combo_lower in map(str.lower, self.known_skills):
                cleaned_skill = self.clean_skill(combo)
                matched_skills.add(cleaned_skill)

        return sorted(list(matched_skills))

    def verify_keyword(self, keyword: str) -> Dict[str, Any]:
        """Enhanced database-aware keyword verification"""
        if not keyword:
            return self._empty_verification_result()
            
        # Clean and normalize keyword
        cleaned_keyword = self.clean_skill(keyword)
        keyword_parts = word_tokenize(cleaned_keyword.lower())
        
        # Initialize result tracking
        found_skills = set()
        found_titles = set()
        
        # Check combinations of words
        for i in range(len(keyword_parts)):
            # Single word
            single = keyword_parts[i]
            self._check_database_match(single, found_skills, found_titles)
            
            # Two-word combinations
            if i < len(keyword_parts) - 1:
                two_words = f"{keyword_parts[i]} {keyword_parts[i+1]}"
                self._check_database_match(two_words, found_skills, found_titles)
            
            # Three-word combinations
            if i < len(keyword_parts) - 2:
                three_words = f"{keyword_parts[i]} {keyword_parts[i+1]} {keyword_parts[i+2]}"
                self._check_database_match(three_words, found_skills, found_titles)

        if found_skills or found_titles:
            return {
                'exists': True,
                'matches': sorted(list(found_skills | found_titles)),
                'skills': sorted(list(found_skills)),
                'job_titles': sorted(list(found_titles)),
                'type': 'skill' if found_skills else 'job_title'
            }

        # No matches - find similar terms from database
        similar_terms = self._find_database_similar_terms(cleaned_keyword)
        return {
            'exists': False,
            'similar_terms': similar_terms,
            'type': None,
            'matches': [],
            'skills': [],
            'job_titles': []
        }

    def _check_database_match(self, term: str, found_skills: set, found_titles: set) -> None:
        """Check term against database entries"""
        term_lower = term.lower()
        
        # Check actual freelancer skills
        for skill in self.known_skills:
            if (term_lower == skill.lower() or 
                term_lower in skill.lower() or 
                skill.lower() in term_lower):
                found_skills.add(self.clean_skill(skill))
        
        # Check actual job titles
        for title in self.job_titles:
            if (term_lower == title.lower() or 
                term_lower in title.lower() or 
                title.lower() in term_lower):
                found_titles.add(self.clean_skill(title))

    def _find_database_similar_terms(self, keyword: str) -> List[str]:
        """Find similar terms from actual database entries"""
        similar = set()
        keyword_lower = keyword.lower()
        
        # Check skills and job titles from database
        all_terms = list(self.known_skills) + list(self.job_titles)
        
        for term in all_terms:
            term_lower = term.lower()
            # Check partial matches and similarity ratio
            if (keyword_lower in term_lower or 
                term_lower in keyword_lower or
                SequenceMatcher(None, keyword_lower, term_lower).ratio() > 0.8):
                similar.add(self.clean_skill(term))
                
        return sorted(list(similar))[:5]

    def _empty_verification_result(self) -> Dict[str, Any]:
        """Return empty verification result structure"""
        return {
            'exists': False,
            'similar_terms': [],
            'type': None,
            'matches': [],
            'skills': [],
            'job_titles': []
        }

    def _check_and_add_match(self, term: str, matches: List[tuple]) -> None:
        """Helper method to check terms against skills and job titles"""
        term_lower = term.lower()
        
        # Check skills
        if any(skill.lower() == term_lower for skill in self.known_skills):
            matches.append(('skill', term))
        
        # Check job titles
        if any(title.lower() == term_lower for title in self.job_titles):
            matches.append(('job_title', term))

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
    """Maps to the users table with is_seller=1 and their associated gigs"""
    id: str
    username: str
    name: str
    job_title: str  # maps to gigs.title
    skills: List[str]  # extracted from gigs.description
    experience: int  # derived from account age
    rating: float  # calculated from gigs.total_stars / gigs.star_number
    hourly_rate: float  # maps to gigs.price
    profile_url: str  # maps to users.img
    availability: bool  # derived from active status
    total_sales: int  # maps to gigs.sales
    description: str = ""  # maps to gigs.description

    def profile_text(self) -> str:
        return f"{self.name} - {self.job_title}. {self.description} Skills: {', '.join(self.skills)}"

    def dict(self) -> Dict[str, Any]:
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
            'description': self.description
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

    def train(self, project_data: List[Dict], freelancer_data: List[Freelancer]):
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
    def __init__(self, freelancers: List[Freelancer], projects: List[Project], skill_extractor: SkillsExtract, collaborative_model=None):
        """
        Initialize the matching engine with freelancers, projects, and skill extraction tools.

        Args:
            freelancers (List[Freelancer]): List of freelancer objects.
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


    def interview_and_evaluate(self, freelancer: Freelancer, project: Project) -> Dict:
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
    
    def ask_professional_questions(self, freelancer: Freelancer, project: Project) -> List[str]:
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

    def accept_or_reject_freelancer(self, freelancer: Freelancer, project: Project):
        client_decision = input(f"Do you want to accept {freelancer.username} for project {project.id}? (yes/no): ")
        if client_decision.lower() == 'yes':
            return True
        return False

    def hire_freelancer(self, freelancer: Freelancer):
        print(f"Notification: {freelancer.username} has been hired!")

def normalize_csv(file_path: str, csv_columns: Optional[Dict[str, str]] = None) -> List[Freelancer]:
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
            freelancer = Freelancer(
                id=row[csv_columns['id']],
                username=row[csv_columns['freelancername']],
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
