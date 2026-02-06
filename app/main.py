import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.init_db import init_db
from app.routes.story_routes import router as story_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Storyteller API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include story routes
app.include_router(story_router)


# Initialize database on startup
@app.on_event("startup")
def startup_event():
    logger.info("Initializing database...")
    init_db()


@app.get("/")
def root():
    return {"message": "AI Storyteller API is running"}
