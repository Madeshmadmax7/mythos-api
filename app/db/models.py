from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
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


class Story(Base):
    """A Story represents a chat/conversation - like a GPT chat session"""
    __tablename__ = "stories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    story_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    genre = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="stories")
    messages = relationship("StoryMessage", back_populates="story", cascade="all, delete-orphan", order_by="StoryMessage.order_index")
    hints = relationship("StoryHint", back_populates="story", cascade="all, delete-orphan")


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


class StoryHint(Base):
    """Accumulated hints for RAG/few-shot - 5-10 words each"""
    __tablename__ = "story_hints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    story_id = Column(Integer, ForeignKey("stories.id"), nullable=False)
    hint_text = Column(String(100), nullable=False)  # Short 5-10 word hint
    message_id = Column(Integer, ForeignKey("story_messages.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    story = relationship("Story", back_populates="hints")
