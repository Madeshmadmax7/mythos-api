import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum, UniqueConstraint
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship

from app.db.connection import Base


class User(Base):
    """User account for authentication"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)  # Hashed password
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    stories = relationship("Story", back_populates="user", cascade="all, delete-orphan")
    reactions = relationship("MessageReaction", back_populates="user", cascade="all, delete-orphan")
    reviews = relationship("MessageReview", back_populates="user", cascade="all, delete-orphan")
    access_requests = relationship("StoryAccess", back_populates="user", cascade="all, delete-orphan")
    change_requests = relationship("StoryChangeRequest", back_populates="user", cascade="all, delete-orphan")


class Story(Base):
    """A Story represents a chat/conversation - like a GPT chat session"""
    __tablename__ = "stories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    hash_id = Column(String(16), unique=True, nullable=False, index=True, default=lambda: uuid.uuid4().hex[:12])
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    story_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    genre = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="stories")
    messages = relationship("StoryMessage", back_populates="story", cascade="all, delete-orphan", order_by="StoryMessage.order_index")
    hints = relationship("StoryHint", back_populates="story", cascade="all, delete-orphan")
    access_requests = relationship("StoryAccess", back_populates="story", cascade="all, delete-orphan")
    change_requests = relationship("StoryChangeRequest", back_populates="story", cascade="all, delete-orphan")


class StoryMessage(Base):
    """Each message in a story chat - user prompt + AI response"""
    __tablename__ = "story_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    story_id = Column(Integer, ForeignKey("stories.id"), nullable=False)
    order_index = Column(Integer, nullable=False)  # Position in chat
    user_prompt = Column(Text, nullable=False)
    ai_response = Column(LONGTEXT, nullable=False)
    hint_context = Column(Text, nullable=True)  # 5-10 word context hint for this segment
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    story = relationship("Story", back_populates="messages")
    reactions = relationship("MessageReaction", back_populates="message", cascade="all, delete-orphan")
    reviews = relationship("MessageReview", back_populates="message", cascade="all, delete-orphan")


class StoryHint(Base):
    """Accumulated hints for RAG/few-shot - 5-10 words each"""
    __tablename__ = "story_hints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    story_id = Column(Integer, ForeignKey("stories.id"), nullable=False)
    hint_text = Column(String(100), nullable=False)  # Short 5-10 word hint
    message_id = Column(Integer, ForeignKey("story_messages.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    story = relationship("Story", back_populates="hints")


class MessageReaction(Base):
    """Like/Dislike reaction for a message - one per user per message"""
    __tablename__ = "message_reactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("story_messages.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    reaction_type = Column(Enum('like', 'dislike', name='reaction_type_enum'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Unique constraint: one reaction per user per message
    __table_args__ = (
        UniqueConstraint('message_id', 'user_id', name='unique_user_message_reaction'),
    )

    message = relationship("StoryMessage", back_populates="reactions")
    user = relationship("User", back_populates="reactions")


class MessageReview(Base):
    """Review/comment for a message - multiple allowed per user per message"""
    __tablename__ = "message_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("story_messages.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    comment = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    message = relationship("StoryMessage", back_populates="reviews")
    user = relationship("User", back_populates="reviews")


class StoryAccess(Base):
    """Tracks permission requests and status for users accessing a story"""
    __tablename__ = "story_access"

    id = Column(Integer, primary_key=True, autoincrement=True)
    story_id = Column(Integer, ForeignKey("stories.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    access_type = Column(Enum('view', 'collaborate', name='access_type_enum'), nullable=False)
    status = Column(Enum('pending', 'approved', 'rejected', name='access_status_enum'), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Constraint: One active request/access per user per story
    __table_args__ = (
        UniqueConstraint('story_id', 'user_id', name='unique_user_story_access'),
    )

    story = relationship("Story", back_populates="access_requests")
    user = relationship("User", back_populates="access_requests")


class StoryChangeRequest(Base):
    """Tracks proposed changes by collaborators"""
    __tablename__ = "story_change_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    story_id = Column(Integer, ForeignKey("stories.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    change_type = Column(Enum('new_message', 'edit', 'refine', name='change_type_enum'), nullable=False)
    target_message_id = Column(Integer, ForeignKey("story_messages.id"), nullable=True) # Null for new_message
    new_content = Column(LONGTEXT, nullable=False) # JSON or Text
    status = Column(Enum('pending', 'approved', 'rejected', name='change_status_enum'), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    story = relationship("Story", back_populates="change_requests")
    user = relationship("User", back_populates="change_requests")
    target_message = relationship("StoryMessage")

