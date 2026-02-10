import os
import uuid
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env variables manually since we are running standalone
from dotenv import load_dotenv
load_dotenv(override=True)

DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "root")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "story_db")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Minimal Story model for just this script to avoid import issues
class Story(Base):
    __tablename__ = "stories"
    id = Column(Integer, primary_key=True)
    hash_id = Column(String(16))
    story_name = Column(String(255))

def fix_hashes():
    db = SessionLocal()
    try:
        # Check if column exists first
        logger.info("Checking if hash_id column exists...")
        from sqlalchemy import text
        try:
            # Try to select the column
            db.execute(text("SELECT hash_id FROM stories LIMIT 1"))
        except Exception:
            logger.info("Column hash_id missing. Adding it now...")
            # Column doesn't exist, add it
            # We use text() for raw SQL
            db.execute(text("ALTER TABLE stories ADD COLUMN hash_id VARCHAR(16) AFTER id"))
            # Add unique index - splitting into separate try/catch in case index exists or fails
            try:
                db.execute(text("CREATE UNIQUE INDEX ix_stories_hash_id ON stories(hash_id)"))
            except Exception as e:
                logger.warning(f"Index creation failed (might already exist): {e}")
            
            db.commit()
            logger.info("Column added successfully.")

        # Find stories with empty or null hash_id
        stories = db.query(Story).filter((Story.hash_id == None) | (Story.hash_id == "")).all()
        
        if not stories:
            logger.info("No stories found with missing hash_ids. Database is clean!")
            return

        logger.info(f"Found {len(stories)} stories with missing hash_ids. Fixing...")
        
        count = 0
        for story in stories:
            new_hash = uuid.uuid4().hex[:12]
            story.hash_id = new_hash
            logger.info(f"Updated story ID {story.id} ({story.story_name}) -> hash: {new_hash}")
            count += 1
            
        db.commit()
        logger.info(f"Successfully backfilled {count} stories with new hash IDs.")
        
    except Exception as e:
        logger.error(f"Error updating database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    fix_hashes()
