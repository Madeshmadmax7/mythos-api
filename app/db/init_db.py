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
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        return False
