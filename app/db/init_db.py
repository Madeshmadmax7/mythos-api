import logging
from app.db.connection import engine, Base

logger = logging.getLogger(__name__)


def init_db():
    """
    Initialize database tables.
    Safe to call even if DB is offline.
    """
    if engine is None:
        logger.warning("Database engine not available, skipping table creation")
        return False
    
    try:
        from app.db import models  # noqa: F401
        Base.metadata.create_all(bind=engine)
        
        # --- Schema Migration ---
        from sqlalchemy import text, inspect
        inspector = inspect(engine)
        columns = [c['name'] for c in inspector.get_columns('stories')]
        
        with engine.connect() as conn:
            # Add 'summary' column if missing
            if 'summary' not in columns:
                logger.info("Migration: Adding 'summary' column to 'stories' table")
                conn.execute(text("ALTER TABLE stories ADD COLUMN summary LONGTEXT NULL"))
                conn.commit()
            
            # Add 'description' column if missing
            if 'description' not in columns:
                logger.info("Migration: Adding 'description' column to 'stories' table")
                conn.execute(text("ALTER TABLE stories ADD COLUMN description TEXT NULL"))
                conn.commit()
            
            # Add 'world_rules' column if missing
            if 'world_rules' not in columns:
                logger.info("Migration: Adding 'world_rules' column to 'stories' table")
                conn.execute(text("ALTER TABLE stories ADD COLUMN world_rules LONGTEXT NULL"))
                conn.commit()
            
            # Migrate story_messages table
            msg_columns = [c['name'] for c in inspector.get_columns('story_messages')]
            
            # Add 'stability_score' column if missing
            if 'stability_score' not in msg_columns:
                logger.info("Migration: Adding 'stability_score' column to 'story_messages' table")
                conn.execute(text("ALTER TABLE story_messages ADD COLUMN stability_score INT NULL"))
                conn.commit()
                    
        # --- Backfill Logic ---
        from sqlalchemy.orm import Session
        from app.db.models import Story
        import uuid
        
        session = Session(bind=engine)
        stories_without_hash = session.query(Story).filter(Story.hash_id == None).all()
        
        if stories_without_hash:
            logger.info(f"Backfilling hash_ids for {len(stories_without_hash)} stories...")
            for story in stories_without_hash:
                story.hash_id = uuid.uuid4().hex[:12]
            session.commit()
            logger.info("Backfill complete")
        session.close()

        logger.info("Database tables and schema checked successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        return False
