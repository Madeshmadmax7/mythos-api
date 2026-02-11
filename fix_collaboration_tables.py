import sys
import os

# Add the parent directory to sys.path so we can import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.connection import engine, Base
from app.db.models import StoryAccess, StoryChangeRequest
from sqlalchemy import inspect

def create_collaboration_tables():
    print("Checking for collaboration tables...")
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    tables_to_create = []
    if "story_access" not in existing_tables:
        tables_to_create.append("story_access")
    if "story_change_requests" not in existing_tables:
        tables_to_create.append("story_change_requests")

    if not tables_to_create:
        print("All collaboration tables already exist.")
        return

    print(f"Creating tables: {', '.join(tables_to_create)}")
    
    # create_all checks individually, but explicit print is nice
    Base.metadata.create_all(bind=engine)
    
    print("Tables created successfully.")

if __name__ == "__main__":
    create_collaboration_tables()
