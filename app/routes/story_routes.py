import json
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db import crud
from app.db.models import User
from app.routes.auth_routes import get_current_user
from app.db.connection import get_db
from app.ai.hints import generate_story_with_context, generate_continuation, refine_single_segment
from app.utils.llm_client import generate_summary, compute_nsi

router = APIRouter(prefix="/api", tags=["Story Chat"])
logger = logging.getLogger(__name__)


# ==================== Request/Response Models ====================

class CreateStoryRequest(BaseModel):
    name: str
    genre: Optional[str] = None


class StoryOut(BaseModel):
    id: int
    user_id: int
    hash_id: str
    story_name: str
    genre: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    message_count: int = 0
    first_prompt: Optional[str] = None
    access_level: Optional[str] = None

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: int
    order_index: int
    user_prompt: str
    ai_response: str
    hint_context: Optional[str]
    stability_score: Optional[int] = None
    created_at: datetime

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
    stability_score: Optional[int] = None
    request_id: Optional[int] = None

class RefineRequest(BaseModel):
    message_id: int
    refine_prompt: str


class RefineResponse(BaseModel):
    message_id: int
    ai_response: str
    hint: str
    request_id: Optional[int] = None


class ContinueRequest(BaseModel):
    story_id: int
    prompt: str


class ContinueResponse(BaseModel):
    message_id: int
    ai_response: str
    hint: str
    stability_score: Optional[int] = None
    request_id: Optional[int] = None


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


# ==================== Collaboration Models ====================

class AccessRequestCreate(BaseModel):
    access_type: str  # 'view' or 'collaborate'

class AccessRequestOut(BaseModel):
    id: int
    story_id: int
    user_id: int
    user_name: str
    access_type: str
    status: str
    created_at: str

    class Config:
        from_attributes = True

class AccessRequestUpdate(BaseModel):
    status: str  # 'approved' or 'rejected'

class ChangeRequestCreate(BaseModel):
    change_type: str  # 'new_message', 'edit', 'refine'
    target_message_id: Optional[int] = None
    new_content: str

class ChangeRequestOut(BaseModel):
    id: int
    story_id: int
    user_id: int
    user_name: str
    change_type: str
    target_message_id: Optional[int]
    new_content: str
    status: str
    created_at: str

    class Config:
        from_attributes = True

class ChangeRequestUpdate(BaseModel):
    status: str # 'approved' or 'rejected'

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
        user_id=story.user_id,
        hash_id=story.hash_id,
        story_name=story.story_name,
        genre=story.genre,
        created_at=story.created_at,
        updated_at=story.updated_at,
        message_count=0,
        access_level="owner"
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
            user_id=story.user_id,
            hash_id=story.hash_id,
            story_name=story.story_name,
            genre=story.genre,
            created_at=story.created_at,
            updated_at=story.updated_at,
            message_count=len(messages),
            first_prompt=first_prompt,
            access_level=crud.check_user_access(db, story.id, current_user.id)
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
    
    # Check access
    access_type = crud.check_user_access(db, story.id, current_user.id)
    if not access_type:
        raise HTTPException(status_code=403, detail="Not authorized to access this story")
    
    messages = crud.get_messages(db, story.id)
    first_prompt = messages[0].user_prompt if messages else None
    
    return StoryOut(
        id=story.id,
        user_id=story.user_id,
        hash_id=story.hash_id,
        story_name=story.story_name,
        genre=story.genre,
        created_at=story.created_at,
        updated_at=story.updated_at,
        message_count=len(messages),
        first_prompt=first_prompt,
        access_level=access_type
    )


@router.get("/stories/hash/{hash_id}", response_model=StoryOut)
def get_story_by_hash(
    hash_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a single story by hash ID."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story_by_hash(db, hash_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Check access
    access_type = crud.check_user_access(db, story.id, current_user.id)
    if not access_type:
        raise HTTPException(status_code=403, detail="Not authorized to access this story")
    
    messages = crud.get_messages(db, story.id)
    first_prompt = messages[0].user_prompt if messages else None
    
    return StoryOut(
        id=story.id,
        user_id=story.user_id,
        hash_id=story.hash_id,
        story_name=story.story_name,
        genre=story.genre,
        created_at=story.created_at,
        updated_at=story.updated_at,
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
    
    # Check access
    access_type = crud.check_user_access(db, story.id, current_user.id)
    if not access_type:
        raise HTTPException(status_code=403, detail="Not authorized to access this story")
    
    messages = crud.get_messages(db, story_id)
    
    return [
        MessageOut(
            id=m.id,
            order_index=m.order_index,
            user_prompt=m.user_prompt,
            ai_response=m.ai_response,
            hint_context=m.hint_context,
            stability_score=m.stability_score,
            created_at=m.created_at
        )
        for m in messages
    ]


@router.put("/messages/{message_id}")
def edit_message(
    message_id: int, 
    request: EditMessageRequest, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Directly edit a message's AI response content."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    message = crud.get_message(db, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Check access
    access_type = crud.check_user_access(db, message.story_id, current_user.id)
    if access_type not in ['owner', 'collaborate']:
        raise HTTPException(status_code=403, detail="Not authorized to edit this message")

    if access_type == 'collaborate':
        # Save as change request (proposal)
        change_req = crud.create_change_request(
            db,
            story_id=message.story_id,
            user_id=current_user.id,
            change_type='edit',
            new_content=request.content,
            target_message_id=message_id
        )
        
        if not change_req:
            raise HTTPException(status_code=500, detail="Failed to save proposal")
        
        return {
            "message_id": message_id,
            "ai_response": request.content,
            "hint_context": message.hint_context,
            "request_id": change_req.id
        }

    # Update message for owner
    updated = crud.update_message(db, message_id, request.content, message.hint_context)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update message")
    
    return {
        "message_id": updated.id,
        "ai_response": updated.ai_response,
        "hint_context": updated.hint_context
    }


def trigger_periodic_summary(db: Session, story_id: int):
    """
    Check if a new summary should be generated (e.g., every 5 messages).
    """
    try:
        messages = crud.get_messages(db, story_id)
        msg_count = len(messages)
        
        # Every 5 messages, update the summary
        if msg_count > 0 and msg_count % 5 == 0:
            logger.info(f"Triggering periodic summarization for story {story_id} (count: {msg_count})")
            current_summary = crud.get_story_summary(db, story_id)
            
            # Use last 10 messages for the 'recent events' to update the summary
            recent_context = []
            for m in messages[-10:]:
                recent_context.append({"role": "user", "content": m.user_prompt})
                recent_context.append({"role": "assistant", "content": m.ai_response})
            
            new_summary = generate_summary(recent_context, current_summary)
            crud.update_story_summary(db, story_id, new_summary)
            logger.info(f"Summary updated for story {story_id}")
    except Exception as e:
        logger.error(f"Error in periodic summarization: {e}")


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
    
    # Check access (require ownership or collaborate access)
    access_type = crud.check_user_access(db, story.id, current_user.id)
    if access_type not in ['owner', 'collaborate']:
        raise HTTPException(status_code=403, detail="Not authorized to generate content for this story")
    
    # Fetch story context
    story_summary = crud.get_story_summary(db, request.story_id)
    story_world_rules = crud.get_world_rules(db, request.story_id)
    existing_messages = crud.get_messages(db, request.story_id)
    previous_hints = [m.hint_context for m in existing_messages if m.hint_context]
    
    # Fetch previous NSI for adaptive injection
    last_message = existing_messages[-1] if existing_messages else None
    previous_nsi = last_message.stability_score if last_message and last_message.stability_score is not None else 100
    
    # Build SLIDING WINDOW chat history (Last 10 messages)
    history = []
    recent_messages = existing_messages[-10:]
    for m in recent_messages:
        history.append({"role": "user", "content": m.user_prompt})
        history.append({"role": "assistant", "content": m.ai_response})
    
    # Determine genre
    genre = request.genre or story.genre or ""
    
    try:
        if len(existing_messages) == 0:
            # First message
            ai_response, new_hint, violations, updated_rules = generate_story_with_context(
                user_prompt=request.prompt,
                genre=genre,
                history=None,
                summary=None,
                previous_hints=None,
                previous_nsi=previous_nsi,
                world_rules=story_world_rules
            )
        else:
            # Continuation - pass history window, summary, and hints
            ai_response, new_hint, violations, updated_rules = generate_continuation(
                user_prompt=request.prompt,
                genre=genre,
                history=history,
                summary=story_summary,
                all_previous_hints=previous_hints,
                previous_nsi=previous_nsi,
                world_rules=story_world_rules
            )
        
        if access_type == 'collaborate':
            # Save as change request (proposal)
            change_req = crud.create_change_request(
                db,
                story_id=request.story_id,
                user_id=current_user.id,
                change_type='new_message',
                new_content=json.dumps({
                    "user_prompt": request.prompt,
                    "ai_response": ai_response,
                    "hint_context": new_hint
                })
            )
            
            if not change_req:
                raise HTTPException(status_code=500, detail="Failed to save proposal")
            
            return GenerateResponse(
                message_id=0, # No message yet
                ai_response=ai_response,
                hint=new_hint or "",
                request_id=change_req.id
            )

        # Compute deterministic NSI from violation counts
        stability_score = compute_nsi(violations)

        # Persist updated world rules
        if updated_rules:
            crud.update_world_rules(db, request.story_id, updated_rules)

        # Save the message for owners
        message = crud.create_message(
            db,
            story_id=request.story_id,
            user_prompt=request.prompt,
            ai_response=ai_response,
            hint_context=new_hint,
            stability_score=stability_score
        )
        
        if not message:
            raise HTTPException(status_code=500, detail="Failed to save message")
        
        # Also save the hint separately for RAG
        if new_hint:
            crud.create_hint(db, request.story_id, new_hint, message.id)
        
        # Trigger periodic summarization
        trigger_periodic_summary(db, request.story_id)
        
        return GenerateResponse(
            message_id=message.id,
            ai_response=ai_response,
            hint=new_hint or "",
            stability_score=stability_score
        )
        
    except Exception as e:
        logger.error(f"Error generating story: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refine", response_model=RefineResponse)
def refine_message(
    request: RefineRequest, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Refine ONLY a specific message. Does not affect other messages.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    message = crud.get_message(db, request.message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Check access
    access_type = crud.check_user_access(db, message.story_id, current_user.id)
    if access_type not in ['owner', 'collaborate']:
        raise HTTPException(status_code=403, detail="Not authorized to refine this story")

    # Build context for refinement
    story_id = message.story_id
    story_summary = crud.get_story_summary(db, story_id)
    story_world_rules = crud.get_world_rules(db, story_id)
    previous_messages = crud.get_previous_messages(db, story_id, message.order_index)
    previous_hints = [m.hint_context for m in previous_messages if m.hint_context]
    
    # Fetch previous NSI for adaptive injection
    last_prev = previous_messages[-1] if previous_messages else None
    previous_nsi = last_prev.stability_score if last_prev and last_prev.stability_score is not None else 100
    
    # Build SLIDING WINDOW context window (Last 10 messages before this one)
    history = []
    recent_prev = previous_messages[-10:]
    for m in recent_prev:
        history.append({"role": "user", "content": m.user_prompt})
        history.append({"role": "assistant", "content": m.ai_response})
    
    try:
        # Refine with hybrid memory context
        refined_text, new_hint, _violations, updated_rules = refine_single_segment(
            original_text=message.ai_response,
            refine_prompt=request.refine_prompt,
            history=history,
            summary=story_summary,
            previous_hints=previous_hints,
            previous_nsi=previous_nsi,
            world_rules=story_world_rules
        )
        
        if access_type == 'collaborate':
            # Save as change request (proposal)
            change_req = crud.create_change_request(
                db,
                story_id=story_id,
                user_id=current_user.id,
                change_type='refine',
                new_content=refined_text,
                target_message_id=request.message_id
            )
            
            if not change_req:
                raise HTTPException(status_code=500, detail="Failed to save proposal")
            
            return RefineResponse(
                message_id=request.message_id,
                ai_response=refined_text,
                hint=new_hint or "",
                request_id=change_req.id
            )

        # Update the message in place for owner
        updated = crud.update_message(
            db,
            message_id=request.message_id,
            ai_response=refined_text,
            hint_context=new_hint
        )
        
        if not updated:
            raise HTTPException(status_code=500, detail="Failed to update message")
        
        return RefineResponse(
            message_id=updated.id,
            ai_response=updated.ai_response,
            hint=updated.hint_context or ""
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
    
    # Check access (require ownership or collaborate access)
    access_type = crud.check_user_access(db, story.id, current_user.id)
    if access_type not in ['owner', 'collaborate']:
        raise HTTPException(status_code=403, detail="Not authorized to continue this story")
    
    # Fetch story context
    story_summary = crud.get_story_summary(db, request.story_id)
    story_world_rules = crud.get_world_rules(db, request.story_id)
    existing_messages = crud.get_messages(db, request.story_id)
    all_hints = [m.hint_context for m in existing_messages if m.hint_context]
    
    # Fetch previous NSI for adaptive injection
    last_message = existing_messages[-1] if existing_messages else None
    previous_nsi = last_message.stability_score if last_message and last_message.stability_score is not None else 100
    
    # Build SLIDING WINDOW chat history (Last 10 messages)
    history = []
    recent_messages = existing_messages[-10:]
    for m in recent_messages:
        history.append({"role": "user", "content": m.user_prompt})
        history.append({"role": "assistant", "content": m.ai_response})
    
    if len(existing_messages) == 0:
        raise HTTPException(status_code=400, detail="Cannot continue - no messages yet. Use /generate first.")
    
    try:
        # Generate continuation with hybrid memory (summary + hints + history window)
        ai_response, new_hint, violations, updated_rules = generate_continuation(
            user_prompt=request.prompt,
            genre=story.genre or "",
            history=history,
            summary=story_summary,
            all_previous_hints=all_hints,
            previous_nsi=previous_nsi,
            world_rules=story_world_rules
        )
        
        if access_type == 'collaborate':
            # Save as change request (proposal)
            change_req = crud.create_change_request(
                db,
                story_id=request.story_id,
                user_id=current_user.id,
                change_type='new_message',
                new_content=json.dumps({
                    "user_prompt": request.prompt,
                    "ai_response": ai_response,
                    "hint_context": new_hint
                })
            )
            
            if not change_req:
                raise HTTPException(status_code=500, detail="Failed to save proposal")
            
            return ContinueResponse(
                message_id=0,
                ai_response=ai_response,
                hint=new_hint or "",
                request_id=change_req.id
            )

        # Compute deterministic NSI from violation counts
        stability_score = compute_nsi(violations)

        # Persist updated world rules
        if updated_rules:
            crud.update_world_rules(db, request.story_id, updated_rules)

        message = crud.create_message(
            db,
            story_id=request.story_id,
            user_prompt=request.prompt,
            ai_response=ai_response,
            hint_context=new_hint,
            stability_score=stability_score
        )
        
        if not message:
            raise HTTPException(status_code=500, detail="Failed to save message")
        
        if new_hint:
            crud.create_hint(db, request.story_id, new_hint, message.id)
        
        # Trigger periodic summarization
        trigger_periodic_summary(db, request.story_id)
        
        return ContinueResponse(
            message_id=message.id,
            ai_response=ai_response,
            hint=new_hint or "",
            stability_score=stability_score
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
    
    # Check access
    access_type = crud.check_user_access(db, message.story_id, current_user.id)
    if access_type not in ['owner', 'collaborate']:
        raise HTTPException(status_code=403, detail="Not authorized to react to this story")

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
    
    # Verify message exists first
    message = crud.get_message(db, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
        
    # Check access
    access_type = crud.check_user_access(db, message.story_id, current_user.id)
    if access_type not in ['owner', 'collaborate']:
        raise HTTPException(status_code=403, detail="Not authorized to review this story")

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

# ==================== Collaboration Endpoints ====================

@router.post("/stories/hash/{hash_id}/request_access", response_model=AccessRequestOut)
def request_access(
    hash_id: str,
    request: AccessRequestCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Request view or collaborate access to a story."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story_by_hash(db, hash_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Don't allow owner to request access
    if story.user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Owner already has access")
    
    access_request = crud.create_access_request(db, story.id, current_user.id, request.access_type)
    if not access_request:
        raise HTTPException(status_code=500, detail="Failed to create access request")
    
    return AccessRequestOut(
        id=access_request.id,
        story_id=access_request.story_id,
        user_id=access_request.user_id,
        user_name=current_user.name,
        access_type=access_request.access_type,
        status=access_request.status,
        created_at=access_request.created_at.isoformat()
    )

@router.get("/stories/hash/{hash_id}/access_requests", response_model=List[AccessRequestOut])
def get_access_requests(
    hash_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get pending access requests (Owner only)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story_by_hash(db, hash_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    access_type = crud.check_user_access(db, story.id, current_user.id)
    if story.user_id != current_user.id and access_type != 'collaborate':
        raise HTTPException(status_code=403, detail="Only owner and collaborators can view requests")
    
    requests = crud.get_story_access_requests(db, story.id)
    
    return [
        AccessRequestOut(
            id=r.id,
            story_id=r.story_id,
            user_id=r.user_id,
            user_name=r.user.name,
            access_type=r.access_type,
            status=r.status,
            created_at=r.created_at.isoformat()
        )
        for r in requests
    ]

@router.put("/stories/hash/{hash_id}/access_requests/{request_id}", response_model=AccessRequestOut)
def update_access_request(
    hash_id: str,
    request_id: int,
    update: AccessRequestUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Approve or Reject access request (Owner only)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story_by_hash(db, hash_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    if story.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only owner can manage requests")
    
    updated_request = crud.update_access_request_status(db, request_id, update.status)
    if not updated_request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return AccessRequestOut(
        id=updated_request.id,
        story_id=updated_request.story_id,
        user_id=updated_request.user_id,
        user_name=updated_request.user.name,
        access_type=updated_request.access_type,
        status=updated_request.status,
        created_at=updated_request.created_at.isoformat()
    )


@router.delete("/stories/hash/{hash_id}/access/{user_id}")
def remove_access(
    hash_id: str,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove a user's access (Owner only, or self to leave)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story_by_hash(db, hash_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Only owner can remove others, or user can remove themselves
    if story.user_id != current_user.id and user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to remove this access")
    
    success = crud.remove_story_access(db, story.id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Access record not found")
    
    return {"message": "Access removed successfully"}

@router.post("/stories/hash/{hash_id}/propose_change", response_model=ChangeRequestOut)
def propose_change(
    hash_id: str,
    request: ChangeRequestCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Propose a change (collaborator only)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story_by_hash(db, hash_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    # Check if user has collaborator access
    access_type = crud.check_user_access(db, story.id, current_user.id)
    if access_type != 'collaborate' and story.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Must be a collaborator to propose changes")
    
    change_request = crud.create_change_request(
        db, 
        story.id, 
        current_user.id, 
        request.change_type, 
        request.new_content, 
        request.target_message_id
    )
    
    if not change_request:
        raise HTTPException(status_code=500, detail="Failed to create change request")
    
    return ChangeRequestOut(
        id=change_request.id,
        story_id=change_request.story_id,
        user_id=change_request.user_id,
        user_name=current_user.name,
        change_type=change_request.change_type,
        target_message_id=change_request.target_message_id,
        new_content=change_request.new_content,
        status=change_request.status,
        created_at=change_request.created_at.isoformat()
    )

@router.get("/stories/hash/{hash_id}/change_requests", response_model=List[ChangeRequestOut])
def get_change_requests(
    hash_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get pending change requests (Owner only)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story_by_hash(db, hash_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    access_type = crud.check_user_access(db, story.id, current_user.id)
    if story.user_id != current_user.id and access_type != 'collaborate':
        raise HTTPException(status_code=403, detail="Only owner and collaborators can view change requests")
    
    requests = crud.get_change_requests(db, story.id)
    
    return [
        ChangeRequestOut(
            id=r.id,
            story_id=r.story_id,
            user_id=r.user_id,
            user_name=r.user.name,
            change_type=r.change_type,
            target_message_id=r.target_message_id,
            new_content=r.new_content,
            status=r.status,
            created_at=r.created_at.isoformat()
        )
        for r in requests
    ]

@router.put("/stories/hash/{hash_id}/change_requests/{request_id}", response_model=ChangeRequestOut)
def update_change_request(
    hash_id: str,
    request_id: int,
    update: ChangeRequestUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Approve or Reject change request (Owner only)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    story = crud.get_story_by_hash(db, hash_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    
    if story.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only owner can manage change requests")
    
    # Update status
    updated_request = crud.update_change_request_status(db, request_id, update.status)
    if not updated_request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # If approved, apply the change!
    if update.status == 'approved':
        if updated_request.change_type == 'new_message':
            # Create message directly
            # Note: new_content likely needs parsing if it's complex, 
            # but for now assume it's just user_prompt or similar? 
            # Actually for 'new_message', we probably need ai_response too.
            # For this MVP, let's assume 'new_content' contains JSON with prompt/response 
            # OR we just treat it as a prompt and generate? 
            # Wait, the user said "collaborator can change story... including refining, generating... needs approval"
            # So the collaborator generates it, sees it pending, and then owner approves.
            # This means we need to store the FULL generated message in 'new_content'.
            import json
            try:
                content_data = json.loads(updated_request.new_content)
                crud.create_message(
                    db, 
                    story.id, 
                    content_data.get('user_prompt', ''), 
                    content_data.get('ai_response', ''),
                    content_data.get('hint_context', '')
                )
                # Trigger periodic summarization after approval
                trigger_periodic_summary(db, story.id)
            except:
                 # Fallback if text
                 pass

        elif updated_request.change_type == 'edit':
             # Apply edit
             crud.update_message(db, updated_request.target_message_id, updated_request.new_content)
        
        elif updated_request.change_type == 'refine':
             # Apply refine
             crud.update_message(db, updated_request.target_message_id, updated_request.new_content)

    return ChangeRequestOut(
        id=updated_request.id,
        story_id=updated_request.story_id,
        user_id=updated_request.user_id,
        user_name=updated_request.user.name,
        change_type=updated_request.access_type if hasattr(updated_request, 'access_type') else updated_request.change_type,
        target_message_id=updated_request.target_message_id,
        new_content=updated_request.new_content,
        status=updated_request.status,
        created_at=updated_request.created_at.isoformat()
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

