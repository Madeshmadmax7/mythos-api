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
        
        # Backfill hashes for existing stories
        # This is a simple migration step
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

        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        return False
