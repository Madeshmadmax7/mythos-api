import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.connection import get_db
from app.db import crud
from app.db.models import User
from app.ai.hints import generate_story_with_context, refine_single_segment, generate_continuation
from app.routes.auth_routes import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Story Chat"])


# ==================== Request/Response Models ====================

class CreateStoryRequest(BaseModel):
    name: str
    genre: Optional[str] = None


class StoryOut(BaseModel):
    id: int
    story_name: str
    genre: Optional[str]
    created_at: str
    updated_at: str
    message_count: int = 0
    first_prompt: Optional[str] = None

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: int
    order_index: int
    user_prompt: str
    ai_response: str
    hint_context: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class GenerateRequest(BaseModel):
    story_id: int
    prompt: str
    genre: Optional[str] = None


class GenerateResponse(BaseModel):
    message_id: int
    ai_response: str
    hint: str

class RefineRequest(BaseModel):
    message_id: int
    refine_prompt: str


class RefineResponse(BaseModel):
    message_id: int
    ai_response: str
    hint: str


class ContinueRequest(BaseModel):
    story_id: int
    prompt: str


class ContinueResponse(BaseModel):
    message_id: int
    ai_response: str
    hint: str


class UpdateStoryRequest(BaseModel):
    name: str


class EditMessageRequest(BaseModel):
    content: str


class ReactionRequest(BaseModel):
    reaction_type: Optional[str] = None  # 'like', 'dislike', or None to remove


class ReactionResponse(BaseModel):
    message_id: int
    reaction_type: Optional[str]
    likes: int
    dislikes: int


class ReviewRequest(BaseModel):
    comment: str


class ReviewOut(BaseModel):
    id: int
    message_id: int
    user_id: int
    user_name: str
    comment: str
    created_at: str

    class Config:
        from_attributes = True


# ==================== Story (Chat) Endpoints ====================

@router.post("/stories", response_model=StoryOut)
def create_story(
    request: CreateStoryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new story/chat."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.create_story(
        db,
        user_id=current_user.id,
        name=request.name,
        genre=request.genre
    )
    
    if not story:
        raise HTTPException(status_code=500, detail="Failed to create story")
    
    return StoryOut(
        id=story.id,
        story_name=story.story_name,
        genre=story.genre,
        created_at=story.created_at.isoformat(),
        updated_at=story.updated_at.isoformat(),
        message_count=0
    )


@router.get("/stories", response_model=List[StoryOut])
def get_stories(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all stories/chats for the current user."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    stories = crud.get_all_stories(db, user_id=current_user.id)
    result = []
    
    for story in stories:
        messages = crud.get_messages(db, story.id)
        first_prompt = messages[0].user_prompt if messages else None
        
        result.append(StoryOut(
            id=story.id,
            story_name=story.story_name,
            genre=story.genre,
            created_at=story.created_at.isoformat(),
            updated_at=story.updated_at.isoformat(),
            message_count=len(messages),
            first_prompt=first_prompt
        ))
    
    return result


@router.get("/stories/{story_id}", response_model=StoryOut)
def get_story(
    story_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a single story."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story(db, story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Verify ownership
    if story.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this story")
    
    messages = crud.get_messages(db, story.id)
    first_prompt = messages[0].user_prompt if messages else None
    
    return StoryOut(
        id=story.id,
        story_name=story.story_name,
        genre=story.genre,
        created_at=story.created_at.isoformat(),
        updated_at=story.updated_at.isoformat(),
        message_count=len(messages),
        first_prompt=first_prompt
    )


@router.delete("/stories/{story_id}")
def delete_story(
    story_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a story and all its messages."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story(db, story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Verify ownership
    if story.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this story")
    
    success = crud.delete_story(db, story_id)
    if not success:
        raise HTTPException(status_code=404, detail="Story not found")
    
    return {"message": "Story deleted successfully"}


@router.put("/stories/{story_id}")
def update_story(
    story_id: int,
    request: CreateStoryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update story name/genre."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story(db, story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Verify ownership
    if story.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this story")
    
    story = crud.update_story(db, story_id, name=request.name, genre=request.genre)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    return {"message": "Story updated"}


# ==================== Message Endpoints ====================

@router.get("/stories/{story_id}/messages", response_model=List[MessageOut])
def get_messages(
    story_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all messages for a story."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story(db, story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Verify ownership
    if story.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this story")
    
    messages = crud.get_messages(db, story_id)
    
    return [
        MessageOut(
            id=m.id,
            order_index=m.order_index,
            user_prompt=m.user_prompt,
            ai_response=m.ai_response,
            hint_context=m.hint_context,
            created_at=m.created_at.isoformat()
        )
        for m in messages
    ]


@router.put("/messages/{message_id}")
def edit_message(message_id: int, request: EditMessageRequest, db: Session = Depends(get_db)):
    """Directly edit a message's AI response content."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    message = crud.get_message(db, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Update message with new content (keep existing hint)
    updated = crud.update_message(db, message_id, request.content, message.hint_context)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update message")
    
    return {
        "message_id": updated.id,
        "ai_response": updated.ai_response,
        "hint_context": updated.hint_context
    }


@router.post("/generate", response_model=GenerateResponse)
def generate_story_message(
    request: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate a new story message.
    For the first message in a story, or continuation.
    Uses accumulated hints as RAG context.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story(db, request.story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Verify ownership
    if story.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this story")
    
    # Get existing messages and their hints for context
    existing_messages = crud.get_messages(db, request.story_id)
    previous_hints = [m.hint_context for m in existing_messages if m.hint_context]
    
    # Determine genre
    genre = request.genre or story.genre or ""
    
    try:
        if len(existing_messages) == 0:
            # First message - generate initial story
            ai_response, new_hint = generate_story_with_context(
                user_prompt=request.prompt,
                genre=genre,
                previous_hints=None,
                previous_content_summary=None
            )
        else:
            # Continuation - use hints to avoid repetition
            ai_response, new_hint = generate_continuation(
                user_prompt=request.prompt,
                genre=genre,
                all_previous_hints=previous_hints
            )
        
        # Save the message
        message = crud.create_message(
            db,
            story_id=request.story_id,
            user_prompt=request.prompt,
            ai_response=ai_response,
            hint_context=new_hint
        )
        
        if not message:
            raise HTTPException(status_code=500, detail="Failed to save message")
        
        # Also save the hint separately for RAG
        if new_hint:
            crud.create_hint(db, request.story_id, new_hint, message.id)
        
        return GenerateResponse(
            message_id=message.id,
            ai_response=ai_response,
            hint=new_hint or ""
        )
        
    except Exception as e:
        logger.error(f"Error generating story: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refine", response_model=RefineResponse)
def refine_message(request: RefineRequest, db: Session = Depends(get_db)):
    """
    Refine ONLY a specific message. Does not affect other messages.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    message = crud.get_message(db, request.message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Get hints from messages BEFORE this one for context
    story_id = message.story_id
    previous_messages = crud.get_previous_messages(db, story_id, message.order_index)
    previous_hints = [m.hint_context for m in previous_messages if m.hint_context]
    
    try:
        # Refine only this segment
        refined_text, new_hint = refine_single_segment(
            original_text=message.ai_response,
            refine_prompt=request.refine_prompt,
            previous_hints=previous_hints
        )
        
        # Update the message in place
        updated = crud.update_message(
            db,
            message_id=request.message_id,
            ai_response=refined_text,
            hint_context=new_hint
        )
        
        if not updated:
            raise HTTPException(status_code=500, detail="Failed to update message")
        
        return RefineResponse(
            message_id=request.message_id,
            ai_response=refined_text,
            hint=new_hint or ""
        )
        
    except Exception as e:
        logger.error(f"Error refining message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/continue", response_model=ContinueResponse)
def continue_story(
    request: ContinueRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Continue the story with a new segment.
    Uses all previous hints to maintain context and avoid repetition.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story(db, request.story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Verify ownership
    if story.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this story")
    
    # Get all existing hints for context
    existing_messages = crud.get_messages(db, request.story_id)
    all_hints = [m.hint_context for m in existing_messages if m.hint_context]
    
    if len(existing_messages) == 0:
        raise HTTPException(status_code=400, detail="Cannot continue - no messages yet. Use /generate first.")
    
    try:
        ai_response, new_hint = generate_continuation(
            user_prompt=request.prompt,
            genre=story.genre or "",
            all_previous_hints=all_hints
        )
        
        message = crud.create_message(
            db,
            story_id=request.story_id,
            user_prompt=request.prompt,
            ai_response=ai_response,
            hint_context=new_hint
        )
        
        if not message:
            raise HTTPException(status_code=500, detail="Failed to save message")
        
        if new_hint:
            crud.create_hint(db, request.story_id, new_hint, message.id)
        
        return ContinueResponse(
            message_id=message.id,
            ai_response=ai_response,
            hint=new_hint or ""
        )
        
    except Exception as e:
        logger.error(f"Error continuing story: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Hints Endpoint ====================

@router.get("/stories/{story_id}/hints")
def get_story_hints(story_id: int, db: Session = Depends(get_db)):
    """Get all accumulated hints for a story (for debugging/display)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    hints = crud.get_hints(db, story_id)
    return [{"id": h.id, "hint": h.hint_text, "message_id": h.message_id} for h in hints]


# ==================== Reaction Endpoints ====================

@router.post("/messages/{message_id}/reaction", response_model=ReactionResponse)
def set_message_reaction(
    message_id: int,
    request: ReactionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Set or update reaction for a message (like/dislike/none)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Validate reaction type
    if request.reaction_type not in [None, 'like', 'dislike']:
        raise HTTPException(status_code=400, detail="Invalid reaction type. Use 'like', 'dislike', or null")
    
    # Verify message exists
    message = crud.get_message(db, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Set the reaction
    crud.set_reaction(db, message_id, current_user.id, request.reaction_type)
    
    # Get current counts
    counts = crud.get_reaction_counts(db, message_id)
    
    return ReactionResponse(
        message_id=message_id,
        reaction_type=request.reaction_type,
        likes=counts["likes"],
        dislikes=counts["dislikes"]
    )


@router.get("/messages/{message_id}/reaction", response_model=ReactionResponse)
def get_message_reaction(
    message_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's reaction and counts for a message."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Get user's reaction
    reaction = crud.get_reaction(db, message_id, current_user.id)
    reaction_type = reaction.reaction_type if reaction else None
    
    # Get counts
    counts = crud.get_reaction_counts(db, message_id)
    
    return ReactionResponse(
        message_id=message_id,
        reaction_type=reaction_type,
        likes=counts["likes"],
        dislikes=counts["dislikes"]
    )


# ==================== Review Endpoints ====================

@router.post("/messages/{message_id}/reviews", response_model=ReviewOut)
def create_message_review(
    message_id: int,
    request: ReviewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a review comment to a message."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
    
    # Verify message exists
    message = crud.get_message(db, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    review = crud.create_review(db, message_id, current_user.id, request.comment)
    if not review:
        raise HTTPException(status_code=500, detail="Failed to create review")
    
    return ReviewOut(
        id=review.id,
        message_id=review.message_id,
        user_id=review.user_id,
        user_name=current_user.name,
        comment=review.comment,
        created_at=review.created_at.isoformat()
    )


@router.get("/messages/{message_id}/reviews", response_model=List[ReviewOut])
def get_message_reviews(
    message_id: int,
    db: Session = Depends(get_db)
):
    """Get all reviews for a message."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    reviews = crud.get_reviews(db, message_id)
    
    return [
        ReviewOut(
            id=r.id,
            message_id=r.message_id,
            user_id=r.user_id,
            user_name=r.user.name if r.user else "Unknown",
            comment=r.comment,
            created_at=r.created_at.isoformat()
        )
        for r in reviews
    ]


@router.delete("/reviews/{review_id}")
def delete_message_review(
    review_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a review (only owner can delete)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    success = crud.delete_review(db, review_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Review not found or not authorized")
    
    return {"message": "Review deleted successfully"}

