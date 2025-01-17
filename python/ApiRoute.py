from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
from pydantic import BaseModel
from sjm import MatchingEngine, Project, SkillsExtract
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ProjectRequest(BaseModel):
    description: str
    required_skills: list[str]
    budget_min: float
    budget_max: float
    complexity: str
    timeline: Optional[int] = None

class InterviewRequest(BaseModel):
    freelancer_id: str
    project_id: str

@app.post("/api/match")
async def match_freelancers(project: ProjectRequest):
    try:
        engine = MatchingEngine()
        project_obj = Project(
            id="temp_id",
            description=project.description,
            required_skills=project.required_skills,
            budget_range=(project.budget_min, project.budget_max),
            complexity=project.complexity,
            timeline=project.timeline
        )
        matches = engine.find_top_matches(project_obj, top_n=5)
        
        # Convert matches to JSON-serializable format
        formatted_matches = [{
            'id': match['freelancer'].id,
            'name': match['freelancer'].name,
            'score': match['combined_score'],
            'hourly_rate': match['freelancer'].hourly_rate,
            'skills': match['freelancer'].skills,
            'rating': match['freelancer'].rating,
            'profile_url': match['freelancer'].profile_url
        } for match in matches]
        
        return {"success": True, "matches": formatted_matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/interview")
async def conduct_interview(request: InterviewRequest):
    try:
        engine = MatchingEngine()
        skill_extractor = SkillsExtract()
        
        # Generate interview questions
        questions = skill_extractor.generate_ai_interview_questions(
            "temp_description",  # You might want to store project descriptions
            ["temp_skill"]  # You might want to store freelancer skills
        )
        
        return {
            "success": True,
            "interview": {
                "freelancer_id": request.freelancer_id,
                "questions": questions
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)