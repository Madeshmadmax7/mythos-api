import logging
import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc
from app.db.models import Story, StoryMessage, StoryHint, MessageReaction, MessageReview, StoryAccess

logger = logging.getLogger(__name__)


# ==================== Story (Chat) Operations ====================

def create_story(db: Session, user_id: int, name: str, genre: str = None, description: str = None) -> Optional[Story]:
    """Create a new story/chat."""
    try:
        story = Story(
            user_id=user_id,
            story_name=name,
            genre=genre,
            description=description,
            hash_id=uuid.uuid4().hex[:12]
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


def get_story_by_hash(db: Session, hash_id: str) -> Optional[Story]:
    """Get a story by its hash_id."""
    try:
        return db.query(Story).filter(Story.hash_id == hash_id).first()
    except Exception as e:
        logger.error(f"Error getting story by hash: {e}")
        return None


def get_all_stories(db: Session, user_id: int = None) -> List[Story]:
    """Get all stories (owned + shared) ordered by most recent."""
    try:
        if not user_id:
            return db.query(Story).order_by(Story.updated_at.desc()).all()
            
        return db.query(Story).outerjoin(StoryAccess).filter(
            or_(
                Story.user_id == user_id,
                and_(
                    StoryAccess.user_id == user_id,
                    StoryAccess.status == 'approved'
                )
            )
        ).order_by(Story.updated_at.desc()).all()
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


# ==================== Reaction Operations ====================

def set_reaction(db: Session, message_id: int, user_id: int, reaction_type: str) -> Optional[MessageReaction]:
    """
    Set or update a reaction for a message.
    reaction_type should be 'like', 'dislike', or None (to remove reaction).
    """
    from app.db.models import MessageReaction
    try:
        # Check if reaction already exists
        existing = db.query(MessageReaction).filter(
            MessageReaction.message_id == message_id,
            MessageReaction.user_id == user_id
        ).first()
        
        if existing:
            if reaction_type is None:
                # Remove the reaction
                db.delete(existing)
                db.commit()
                return None
            else:
                # Update existing reaction
                existing.reaction_type = reaction_type
                db.commit()
                db.refresh(existing)
                return existing
        else:
            if reaction_type is None:
                return None
            # Create new reaction
            reaction = MessageReaction(
                message_id=message_id,
                user_id=user_id,
                reaction_type=reaction_type
            )
            db.add(reaction)
            db.commit()
            db.refresh(reaction)
            return reaction
    except Exception as e:
        logger.error(f"Error setting reaction: {e}")
        db.rollback()
        return None


def get_reaction(db: Session, message_id: int, user_id: int) -> Optional[MessageReaction]:
    """Get user's reaction for a message."""
    from app.db.models import MessageReaction
    try:
        return db.query(MessageReaction).filter(
            MessageReaction.message_id == message_id,
            MessageReaction.user_id == user_id
        ).first()
    except Exception as e:
        logger.error(f"Error getting reaction: {e}")
        return None


def get_reaction_counts(db: Session, message_id: int) -> dict:
    """Get like and dislike counts for a message."""
    from app.db.models import MessageReaction
    try:
        likes = db.query(MessageReaction).filter(
            MessageReaction.message_id == message_id,
            MessageReaction.reaction_type == 'like'
        ).count()
        
        dislikes = db.query(MessageReaction).filter(
            MessageReaction.message_id == message_id,
            MessageReaction.reaction_type == 'dislike'
        ).count()
        
        return {"likes": likes, "dislikes": dislikes}
    except Exception as e:
        logger.error(f"Error getting reaction counts: {e}")
        return {"likes": 0, "dislikes": 0}


# ==================== Review Operations ====================

def create_review(db: Session, message_id: int, user_id: int, comment: str) -> Optional[MessageReview]:
    """Create a review/comment for a message."""
    from app.db.models import MessageReview
    try:
        review = MessageReview(
            message_id=message_id,
            user_id=user_id,
            comment=comment
        )
        db.add(review)
        db.commit()
        db.refresh(review)
        return review
    except Exception as e:
        logger.error(f"Error creating review: {e}")
        db.rollback()
        return None


def get_reviews(db: Session, message_id: int) -> List[MessageReview]:
    """Get all reviews for a message."""
    from app.db.models import MessageReview
    try:
        return db.query(MessageReview).filter(
            MessageReview.message_id == message_id
        ).order_by(MessageReview.created_at.desc()).all()
    except Exception as e:
        logger.error(f"Error getting reviews: {e}")
        return []


def delete_review(db: Session, review_id: int, user_id: int) -> bool:
    """Delete a review (only if owned by user)."""
    from app.db.models import MessageReview
    try:
        review = db.query(MessageReview).filter(
            MessageReview.id == review_id,
            MessageReview.user_id == user_id
        ).first()
        
        if review:
            db.delete(review)
            db.commit()
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting review: {e}")
        db.rollback()
        return False


# ==================== Collaboration - Access Operations ====================

def create_access_request(db: Session, story_id: int, user_id: int, access_type: str) -> Optional[object]:
    """Create a request for viewing or collaborating on a story."""
    from app.db.models import StoryAccess
    try:
        # Check if exists
        existing = db.query(StoryAccess).filter(
            StoryAccess.story_id == story_id,
            StoryAccess.user_id == user_id
        ).first()

        if existing:
            # Update existing if status is not approved, or just return existing
            if existing.status != 'approved':
                existing.access_type = access_type
                existing.status = 'pending'
                db.commit()
                db.refresh(existing)
            return existing
        
        request = StoryAccess(
            story_id=story_id,
            user_id=user_id,
            access_type=access_type,
            status='pending'
        )
        db.add(request)
        db.commit()
        db.refresh(request)
        return request
    except Exception as e:
        logger.error(f"Error creating access request: {e}")
        db.rollback()
        return None

def get_story_access_requests(db: Session, story_id: int) -> List[object]:
    """Get all access requests for a story."""
    from app.db.models import StoryAccess
    try:
        return db.query(StoryAccess).filter(
            StoryAccess.story_id == story_id
        ).all()
    except Exception as e:
        logger.error(f"Error getting access requests: {e}")
        return []

def update_access_request_status(db: Session, request_id: int, status: str) -> Optional[object]:
    """Update status of an access request (approved, rejected)."""
    from app.db.models import StoryAccess
    try:
        request = db.query(StoryAccess).filter(StoryAccess.id == request_id).first()
        if request:
            request.status = status
            db.commit()
            db.refresh(request)
        return request
    except Exception as e:
        logger.error(f"Error updating access request: {e}")
        db.rollback()
        return None

def check_user_access(db: Session, story_id: int, user_id: int) -> Optional[str]:
    """Check if user has access to story. Returns 'view', 'collaborate', or None."""
    from app.db.models import StoryAccess, Story
    try:
        # Owner always has access
        story = db.query(Story).filter(Story.id == story_id).first()
        if story and story.user_id == user_id:
            return 'owner'

        access = db.query(StoryAccess).filter(
            StoryAccess.story_id == story_id,
            StoryAccess.user_id == user_id
        ).first()

        if access:
            if access.status == 'approved':
                return access.access_type
            elif access.status == 'pending':
                return 'pending'
        return None
    except Exception as e:
        logger.error(f"Error checking user access: {e}")
        return None


# ==================== Collaboration - Change Operations ====================

def create_change_request(db: Session, story_id: int, user_id: int, change_type: str, new_content: str, target_message_id: int = None) -> Optional[object]:
    """Propose a change (new message, edit, refine)."""
    from app.db.models import StoryChangeRequest
    try:
        request = StoryChangeRequest(
            story_id=story_id,
            user_id=user_id,
            change_type=change_type,
            new_content=new_content,
            target_message_id=target_message_id,
            status='pending'
        )
        db.add(request)
        db.commit()
        db.refresh(request)
        return request
    except Exception as e:
        logger.error(f"Error creating change request: {e}")
        db.rollback()
        return None

def get_change_requests(db: Session, story_id: int) -> List[object]:
    """Get pending change requests for a story."""
    from app.db.models import StoryChangeRequest
    try:
        return db.query(StoryChangeRequest).filter(
            StoryChangeRequest.story_id == story_id,
            StoryChangeRequest.status == 'pending'
        ).all()
    except Exception as e:
        logger.error(f"Error getting change requests: {e}")
        return []

def update_change_request_status(db: Session, request_id: int, status: str) -> Optional[object]:
    """Update change request status. If approved, caller must apply change manually."""
    from app.db.models import StoryChangeRequest
    try:
        request = db.query(StoryChangeRequest).filter(StoryChangeRequest.id == request_id).first()
        if request:
            request.status = status
            db.commit()
            db.refresh(request)
        return request
    except Exception as e:
        logger.error(f"Error updating change request: {e}")
        return None

def remove_story_access(db: Session, story_id: int, user_id: int) -> bool:
    """Remove a user's access to a story (member or pending)."""
    from app.db.models import StoryAccess
    try:
        access = db.query(StoryAccess).filter(
            StoryAccess.story_id == story_id,
            StoryAccess.user_id == user_id
        ).first()
        
        if access:
            db.delete(access)
            db.commit()
            return True
        return False
    except Exception as e:
        logger.error(f"Error removing story access: {e}")
        db.rollback()
        return False
