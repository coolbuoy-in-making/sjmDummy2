import os
import sys
import uuid
import json
import logging
from typing import List, Dict, Optional
import numpy as np
import nltk
nltk.download('punkt_tab')

from sjm import (
    SkillsExtract, 
    Freelancer, 
    Project, 
    Server, 
    normalize_csv, 
    MatchingEngine,
    CollaborativeModel
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('upwork_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UpworkIntegrationModel:
    def __init__(self, csv_file_path: str):
        """
        Initialize the Upwork Integration Model
        
        Args:
            csv_file_path (str): Path to the Upwork freelancers CSV file
        """
        self.csv_file_path = csv_file_path
        self.skill_extractor = SkillsExtract()
        self.freelancers = None
        self.matching_engine = None
        
        # Upwork-specific weights for the matching system
        self.custom_weights = {
            'content': 0.5,
            'collaborative': 0.4,
            'experience': 0.2,
            'rating': 0.1,
            'top_rated': 0.3, 
        }

    def load_freelancers(self) -> List[Freelancer]:
        """
        Load and normalize freelancers from Upwork CSV.

        Returns:
            List[Freelancer]: Normalized freelancer data.
        """
        try:
            upwork_columns = {
                'id': 'freelancer_id',
                'username': 'name',
                'name': 'name',
                'job_title': 'job_title',
                'skills': 'skills',
                'experience': 'years_of_experience',
                'rating': 'success_rate',
                'hourly_rate': 'hourly_rate',
                'profile_url': 'portfolio_url',
                'total_sales': 'total_jobs',
                'availability': 'top_rated',
                'total_hours': 'hours_worked'
            }

            freelancers = normalize_csv(self.csv_file_path, upwork_columns)

            for freelancer in freelancers:
                freelancer.availability = freelancer.availability == 'True'

            self.freelancers = freelancers
            logger.info(f"Loaded {len(freelancers)} freelancers from {self.csv_file_path}")
            return freelancers
        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.csv_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading freelancers: {e}")
            raise

    def customize_matching_engine(self):
        """
        Customizes the collaborative model for Upwork to include total_jobs and success_rate.
        """
        class UpworkCollaborativeModel(CollaborativeModel):
            def train(self, project_data: List[Dict], freelancer_data: List[Freelancer]):
                """
                Train the collaborative model using total_jobs and success_rate for Upwork.
                """
                self.freelancer_data = freelancer_data
                self.project_data = project_data

                num_freelancers = len(freelancer_data)
                if num_freelancers == 0:
                    self.interaction_matrix = np.zeros((num_freelancers, 2))
                    return

                total_jobs = np.array([freelancer.total_sales for freelancer in freelancer_data])
                success_rates = np.array([freelancer.rating for freelancer in freelancer_data])

                total_jobs_norm = (total_jobs - total_jobs.min()) / (total_jobs.max() - total_jobs.min())
                success_rates_norm = success_rates / 100.0

                self.interaction_matrix = np.column_stack((total_jobs_norm, success_rates_norm))

            def predict(self, project_description: str, project_skills: List[str]) -> List[float]:
                """
                Predict match scores based on collaborative metrics.
                """
                if self.interaction_matrix is None or self.interaction_matrix.size == 0:
                    logger.warning("Interaction matrix is empty. Returning zero scores.")
                    return [0.0] * len(self.freelancer_data)

                scores = np.nanmean(self.interaction_matrix, axis=1)
                return np.nan_to_num(scores).tolist()

        return UpworkCollaborativeModel()

    def adjust_weights_for_project(self, project: Project) -> Dict[str, float]:
        """
        Dynamically adjust weights based on project complexity and Upwork-specific factors.
        """
        base_weights = self.custom_weights.copy()
        if project.complexity == 'high':
            base_weights['content'] += 0.1
            base_weights['collaborative'] += 0.1
            base_weights['top_rated'] += 0.1
        elif project.complexity == 'low':
            base_weights['hourly_rate'] = 0.2
        return base_weights

    def filter_freelancers(self, project: Project, matches: List[Dict]) -> List[Dict]:
        """
        Filter freelancers based on Upwork-specific constraints.
        """
        filtered_matches = []
        for match in matches:
            freelancer = match['freelancer']
            
            # Upwork-specific hard constraints
            if freelancer.hourly_rate < project.budget_range[0] or freelancer.hourly_rate > project.budget_range[1]:
                logger.debug(f"Excluded {freelancer.username}: Hourly rate ${freelancer.hourly_rate} out of budget.")
                continue
            
            # Prioritize top-rated freelancers for critical projects
            if project.complexity == 'high' and not freelancer.availability:
                logger.debug(f"Excluded {freelancer.username}: Not top-rated for high-complexity project.")
                continue
            
            # Refine skill matching and check overlap
            overlap_count = self.matching_engine.refine_skill_matching(project.required_skills, freelancer.skills)
            if overlap_count < 2:  # Require at least 2 overlapping or similar skills
                logger.debug(f"Excluded {freelancer.username}: Insufficient skill overlap ({overlap_count} matching skills).")
                continue
            
            # Passed all filters
            filtered_matches.append(match)
        return filtered_matches

    def find_top_matches(self, project: Project, top_n: int = 5):
        try:
            if not self.matching_engine:
                self.initialize_matching_engine()

            self.custom_weights = self.adjust_weights_for_project(project)

            all_matches = self.matching_engine.match_freelancers(project, weights=self.custom_weights)
            filtered_matches = self.filter_freelancers(project, all_matches)

            top_matches = filtered_matches[:top_n]
            logger.info(f"Found {len(top_matches)} top matches")
            return top_matches
        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            raise
        
    def run_upwork_matching(self):
        """
        Main workflow for Upwork freelancer matching.
        """
        try:
            self.load_freelancers()
            project = self.collect_project_details()
            top_matches = self.find_top_matches(project)

            for match in top_matches:
                freelancer = match['freelancer']
                print(f"\nCandidate: {freelancer.name}")
                print(f"Match Score: {match['combined_score']:.2f}")
                print(f"Job Title: {freelancer.job_title}")
                print(f"Skills: {', '.join(freelancer.skills)}")
                print(f"Hourly Rate: {freelancer.hourly_rate}")
                print(f"Total Jobs: {freelancer.total_sales}")
                print(f"Success Rate: {freelancer.rating}%")

                if input("Interview this freelancer? (yes/no): ").strip().lower() == 'yes':
                    interview_results = self.interview_freelancer(freelancer, project)
                    print(f"Interview Results: {interview_results}")

                    if input("Hire this freelancer? (yes/no): ").strip().lower() == 'yes':
                        logger.info(f"Hired freelancer: {freelancer.username}")
                        break
        except Exception as e:
            logger.error(f"Upwork matching process error: {e}")

    def initialize_matching_engine(self):
        """
        Initialize a Upwork-customized matching engine.
        """
        if not self.freelancers:
            self.load_freelancers()

        collaborative_model = self.customize_matching_engine()

        self.matching_engine = MatchingEngine(
            freelancers=self.freelancers,
            projects=[],
            skill_extractor=self.skill_extractor,
            collaborative_model=collaborative_model,
        )
        self.matching_engine.train_models()
        logger.info("Matching engine initialized with Upwork-specific customization")
        
        
    def interview_freelancer(self, freelancer: Freelancer, project: Project):
        """
        Conduct interview with a selected freelancer.
        """
        logger.info(f"Starting interview for {freelancer.username}")

            # Initialize server
        server = Server()
        server.start_server()

        try:
            questions = []
            # Collect custom client message
            custom_question = input("Do you want to send a custom question to the freelancer? (yes/no): ")
            if custom_question.lower() == "yes":
                cl_questions = input("Enter your question (use commas if you want to ask more than one question): ")
                
                client_questions = [ques.strip() for ques in cl_questions.split(',') if cl_questions.strip]
                
                questions.extend(client_questions)
            else:
                None

            # Create interview context
            interview_context = {
                'project_id': project.id,
                'project_description': project.description,
                'freelancer_username': freelancer.username,
                'freelancer_job_title': freelancer.job_title,
                'freelancer_skills': freelancer.skills,
                'freelancer_id': freelancer.id,
                'hourly_rate': freelancer.hourly_rate,
            }

            # Send context to freelancer
            server.send_message(json.dumps(interview_context, indent=2))

            # Receive interview questions
            questions_json = server.receive_message()
            questions = json.loads(questions_json)
            
            print("Currently interviewing user, you will receive a response shortly....")

            # Wait for freelancer's answers
            answers_json = server.receive_message()
            answers = json.loads(answers_json)

            # Evaluate answers
            score = self.evaluate_answers(answers)

            # Prepare interview results
            interview_results = {
                'freelancer_username': freelancer.username,
                'freelancer_job_title': freelancer.job_title,
                'questions_and_answers': [{"question": q, "answer": a} for q, a in answers.items()],
                'score': score,
                'project_id': project.id
            }
            
            print(f"\nInterview Evaluation Complete for {freelancer.username}!\n")
            
            # Send results back to client
            server.send_message(json.dumps(interview_results, indent=2))

        finally:
            # Close the connection
            server.close_connection()
            print("Server closed after interview.")
        return f"Interview Results:\n{json.dumps(interview_results, indent=2)}"
     
    def collect_project_details(self) -> Project:
        try:
            logger.info("Starting project details collection")

            description = input("Enter Project Description: ").strip()
            inferred_skills = self.skill_extractor.extract_skills(description)
            logger.info(f"AI-extracted skills: {inferred_skills}")

            manual_skills_input = input("Add additional skills (comma-separated, or press Enter to skip): ").strip()
            manual_skills = [skill.strip() for skill in manual_skills_input.split(',')] if manual_skills_input else []
            required_skills = list(set(inferred_skills + manual_skills))

            min_budget = self._get_valid_input("Minimum Budget ($): ", float)
            max_budget = self._get_valid_input("Maximum Budget ($): ", float, lambda x: x >= min_budget)

            complexity = self._get_choice_input("Project Complexity (low/medium/high): ", ['low', 'medium', 'high'])
            timeline = self._get_valid_input("Project Timeline (days): ", int, lambda x: x > 0)

            project = Project(
                id=str(uuid.uuid4()),
                description=description,
                required_skills=required_skills,
                budget_range=(min_budget, max_budget),
                complexity=complexity,
                timeline=timeline,
            )
            logger.info("Project details collected successfully")
            return project
        except Exception as e:
            logger.error(f"Error collecting project details: {e}")
            raise

    def evaluate_answers(self, answers: Dict[str, str]) -> float:
        # Simple evaluation based on response length (ai will replace this)
        score = sum(len(answer) for answer in answers.values()) / 100
        return round(score, 2)

    @staticmethod
    def _get_valid_input(prompt, input_type, condition=lambda x: True):
        while True:
            try:
                value = input_type(input(prompt))
                if not condition(value):
                    raise ValueError("Invalid input.")
                return value
            except ValueError:
                print("Invalid input. Please try again.")

    @staticmethod
    def _get_choice_input(prompt, choices):
        while True:
            choice = input(prompt).strip().lower()
            if choice in choices:
                return choice
            print(f"Invalid choice. Please select from {', '.join(choices)}.")

def main():
    while True: 
        if len(sys.argv) > 1 and sys.argv[1] == "freelancer":
            # Run the freelancer logic
            from freelancer import main as freelancer_main
            freelancer_main("127.0.0.1", "65432")  # Use appropriate host/port
        else:
            print("Starting Upwork Model....")
                # Entering the CSV file path manually
            csv_path = "upwork_freelancers.csv"
            
            if not csv_path:
                print("No file selected. Exiting.")
                return
            
            if not os.path.isfile(csv_path):
                print(f"Error: The file '{csv_path}' does not exist. Please check and try again.")
                return

            try:
                # Initialize the Upwork integration model
                upwork_model = UpworkIntegrationModel(csv_path)
                
                # Run the Upwork matching process
                upwork_model.run_upwork_matching()
            except Exception as e:
                logger.error(f"Upwork integration failed: {e}") 
                  
        should_continue = input("\nWould you like to run the Upwork Model process again? (yes/no): ").strip().lower()
        if should_continue != 'yes':
            print("Exiting program. Goodbye!")
            break 

if __name__ == "__main__":
    main()