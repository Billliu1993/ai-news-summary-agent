"""Data models for Hacker News stories and related objects."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class StoryType(str, Enum):
    """Types of Hacker News items."""
    STORY = "story"
    ASK = "ask"
    SHOW = "show"
    JOB = "job"
    COMMENT = "comment"
    POLL = "poll"
    POLLOPT = "pollopt"


class HNItem(BaseModel):
    """Base model for Hacker News items."""
    
    id: int
    by: Optional[str] = None
    time: int = Field(..., description="Unix timestamp")
    type: StoryType
    dead: bool = False
    deleted: bool = False
    
    @property
    def datetime(self) -> datetime:
        """Convert Unix timestamp to datetime."""
        return datetime.fromtimestamp(self.time)
    
    @property
    def age_hours(self) -> float:
        """Get age in hours."""
        return (datetime.now() - self.datetime).total_seconds() / 3600
    
    @property
    def is_recent(self) -> bool:
        """Check if item is from last 24 hours."""
        return self.age_hours <= 24.0
    
    @property
    def is_valid(self) -> bool:
        """Check if item is valid (not dead or deleted)."""
        return not self.dead and not self.deleted


class Story(HNItem):
    """Represents a Hacker News story."""
    
    title: str
    url: Optional[str] = None
    text: Optional[str] = None
    score: int = 0
    descendants: int = Field(default=0, description="Number of comments")
    kids: List[int] = Field(default_factory=list, description="Comment IDs")
    
    @validator('title')
    def title_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
    
    @property
    def has_url(self) -> bool:
        """Check if story has an external URL."""
        return self.url is not None and self.url.strip() != ""
    
    @property
    def is_text_post(self) -> bool:
        """Check if story is a text-only post."""
        return not self.has_url and bool(self.text)
    
    @property
    def comment_count(self) -> int:
        """Get number of comments."""
        return self.descendants
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.title} ({self.score} points, {self.comment_count} comments)"


class Comment(HNItem):
    """Represents a Hacker News comment."""
    
    text: Optional[str] = None
    parent: int = Field(..., description="Parent item ID")
    kids: List[int] = Field(default_factory=list, description="Reply IDs")
    
    @property
    def has_text(self) -> bool:
        """Check if comment has text content."""
        return self.text is not None and self.text.strip() != ""


class Summary(BaseModel):
    """Represents a summary of stories."""
    
    content: str
    story_count: int = Field(..., gt=0)
    topics: List[str]
    generated_at: datetime = Field(default_factory=datetime.now)
    total_score: int = 0
    
    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Summary content cannot be empty')
        return v.strip()
    
    @validator('topics')
    def topics_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('At least one topic is required')
        return [topic.strip() for topic in v if topic.strip()]
    
    def __str__(self) -> str:
        """String representation."""
        topics_str = ", ".join(self.topics)
        return f"Summary of {self.story_count} stories about {topics_str}"


class SlackMessage(BaseModel):
    """Represents a message to send to Slack."""
    
    text: str
    blocks: Optional[List[dict]] = None
    username: Optional[str] = "HN Bot"
    icon_emoji: Optional[str] = ":newspaper:"
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message text cannot be empty')
        return v.strip()
    
    def to_payload(self) -> dict:
        """Convert to Slack webhook payload."""
        payload = {"text": self.text}
        
        if self.blocks:
            payload["blocks"] = self.blocks
        if self.username:
            payload["username"] = self.username
        if self.icon_emoji:
            payload["icon_emoji"] = self.icon_emoji
            
        return payload