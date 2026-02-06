import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from app.db.models import Story, StoryMessage, StoryHint

logger = logging.getLogger(__name__)


# ==================== Story (Chat) Operations ====================

def create_story(db: Session, name: str, genre: str = None, description: str = None) -> Optional[Story]:
    """Create a new story/chat."""
    try:
        story = Story(
            story_name=name,
            genre=genre,
            description=description
        )
        db.add(story)
        db.commit()
        db.refresh(story)
        return story
    except Exception as e:
        logger.error(f"Error creating story: {e}")
        db.rollback()
        return None


def get_story(db: Session, story_id: int) -> Optional[Story]:
    """Get a story by ID."""
    try:
        return db.query(Story).filter(Story.id == story_id).first()
    except Exception as e:
        logger.error(f"Error getting story: {e}")
        return None


def get_all_stories(db: Session) -> List[Story]:
    """Get all stories ordered by most recent."""
    try:
        return db.query(Story).order_by(Story.updated_at.desc()).all()
    except Exception as e:
        logger.error(f"Error getting stories: {e}")
        return []


def update_story(db: Session, story_id: int, name: str = None, genre: str = None) -> Optional[Story]:
    """Update story name or genre."""
    try:
        story = db.query(Story).filter(Story.id == story_id).first()
        if story:
            if name:
                story.story_name = name
            if genre:
                story.genre = genre
            db.commit()
            db.refresh(story)
        return story
    except Exception as e:
        logger.error(f"Error updating story: {e}")
        db.rollback()
        return None


def delete_story(db: Session, story_id: int) -> bool:
    """Delete a story and all its messages."""
    try:
        story = db.query(Story).filter(Story.id == story_id).first()
        if story:
            db.delete(story)
            db.commit()
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting story: {e}")
        db.rollback()
        return False


# ==================== Message Operations ====================

def create_message(db: Session, story_id: int, user_prompt: str, ai_response: str, hint_context: str = None) -> Optional[StoryMessage]:
    """Create a new message in a story."""
    try:
        # Get next order index
        max_order = db.query(StoryMessage).filter(
            StoryMessage.story_id == story_id
        ).count()
        
        message = StoryMessage(
            story_id=story_id,
            order_index=max_order,
            user_prompt=user_prompt,
            ai_response=ai_response,
            hint_context=hint_context
        )
        db.add(message)
        
        # Update story's updated_at
        story = db.query(Story).filter(Story.id == story_id).first()
        if story:
            from datetime import datetime
            story.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(message)
        return message
    except Exception as e:
        logger.error(f"Error creating message: {e}")
        db.rollback()
        return None


def get_message(db: Session, message_id: int) -> Optional[StoryMessage]:
    """Get a message by ID."""
    try:
        return db.query(StoryMessage).filter(StoryMessage.id == message_id).first()
    except Exception as e:
        logger.error(f"Error getting message: {e}")
        return None


def get_messages(db: Session, story_id: int) -> List[StoryMessage]:
    """Get all messages for a story in order."""
    try:
        return db.query(StoryMessage).filter(
            StoryMessage.story_id == story_id
        ).order_by(StoryMessage.order_index).all()
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        return []


def update_message(db: Session, message_id: int, ai_response: str, hint_context: str = None) -> Optional[StoryMessage]:
    """Update a message's AI response (for refinement)."""
    try:
        message = db.query(StoryMessage).filter(StoryMessage.id == message_id).first()
        if message:
            message.ai_response = ai_response
            if hint_context:
                message.hint_context = hint_context
            from datetime import datetime
            message.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(message)
        return message
    except Exception as e:
        logger.error(f"Error updating message: {e}")
        db.rollback()
        return None


def get_previous_messages(db: Session, story_id: int, before_order: int) -> List[StoryMessage]:
    """Get all messages before a certain order index."""
    try:
        return db.query(StoryMessage).filter(
            StoryMessage.story_id == story_id,
            StoryMessage.order_index < before_order
        ).order_by(StoryMessage.order_index).all()
    except Exception as e:
        logger.error(f"Error getting previous messages: {e}")
        return []


# ==================== Hint Operations ====================

def create_hint(db: Session, story_id: int, hint_text: str, message_id: int = None) -> Optional[StoryHint]:
    """Create a new hint for a story."""
    try:
        hint = StoryHint(
            story_id=story_id,
            hint_text=hint_text[:100],  # Ensure max 100 chars
            message_id=message_id
        )
        db.add(hint)
        db.commit()
        db.refresh(hint)
        return hint
    except Exception as e:
        logger.error(f"Error creating hint: {e}")
        db.rollback()
        return None


def get_hints(db: Session, story_id: int) -> List[StoryHint]:
    """Get all hints for a story."""
    try:
        return db.query(StoryHint).filter(
            StoryHint.story_id == story_id
        ).order_by(StoryHint.created_at).all()
    except Exception as e:
        logger.error(f"Error getting hints: {e}")
        return []


def get_hints_before_message(db: Session, story_id: int, message_id: int) -> List[StoryHint]:
    """Get hints created before a specific message."""
    try:
        message = db.query(StoryMessage).filter(StoryMessage.id == message_id).first()
        if not message:
            return []
        
        return db.query(StoryHint).filter(
            StoryHint.story_id == story_id,
            StoryHint.message_id < message_id
        ).order_by(StoryHint.created_at).all()
    except Exception as e:
        logger.error(f"Error getting hints before message: {e}")
        return []
