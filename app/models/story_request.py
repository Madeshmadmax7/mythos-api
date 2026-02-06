from pydantic import BaseModel
from typing import Optional


class StoryRequest(BaseModel):
    context: str
    genre: Optional[str] = None


class StoryResponse(BaseModel):
    story: str
