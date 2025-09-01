"""Configuration management for the HN Agent using Pydantic settings."""

import re
from functools import lru_cache
from typing import List, Optional, Set

from pydantic import Field, field_validator, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


# Default topics configuration
DEFAULT_TOPICS = [
    "AI", "artificial intelligence", "machine learning", "deep learning",
    "venture capital", "funding", "IPO",
    "data science", "analytics", "big data"
]

# Topic categories for better organization
TOPIC_CATEGORIES = {
    "ai_ml": ["AI", "artificial intelligence", "machine learning", "deep learning", 
              "neural networks", "LLM", "GPT", "OpenAI", "transformers"],
    "funding": ["venture capital", "funding", "IPO", "VC", "acquisition"],
}


class HNAgentSettings(BaseSettings):
    """Pydantic settings for HN Agent configuration."""
    
    # API Keys and Authentication
    openai_api_key: str = Field(..., description="OpenAI API key for summarization")
    slack_webhook_url: HttpUrl = Field(..., description="Slack webhook URL for notifications")
    
    # OpenAI Configuration
    openai_model: str = Field(
        default="gpt-5-nano", 
        description="OpenAI model to use for summarization"
    )
    openai_max_retries: int = Field(
        default=3, 
        ge=1, 
        le=10, 
        description="Maximum retry attempts for OpenAI API"
    )
    openai_temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0, 
        description="OpenAI temperature for summary generation"
    )
    openai_max_tokens: int = Field(
        default=1000, 
        ge=100, 
        le=4000, 
        description="Maximum tokens for summary"
    )
    
    # Slack Configuration
    slack_timeout: int = Field(
        default=30, 
        ge=5, 
        le=120, 
        description="Slack webhook timeout in seconds"
    )
    slack_username: str = Field(
        default="HN Bot", 
        description="Bot username for Slack messages"
    )
    slack_icon_emoji: str = Field(
        default=":newspaper:", 
        description="Bot icon emoji for Slack"
    )
    
    # Topics Configuration
    topics: str = Field(
        default="AI,programming,startups", 
        description="Topics to filter for (comma-separated)"
    )
    enable_all_default_topics: bool = Field(
        default=False, 
        description="Enable all predefined default topics"
    )
    topic_categories: str = Field(
        default="", 
        description="Topic categories to enable (ai_ml, programming, startups, etc.)"
    )
    
    # Story Filtering
    max_stories: int = Field(
        default=50, 
        ge=10, 
        le=200, 
        description="Maximum stories to fetch from HN"
    )
    summary_max_stories: int = Field(
        default=15, 
        ge=5, 
        le=50, 
        description="Maximum stories to include in summary"
    )
    filter_min_score: int = Field(
        default=10, 
        ge=1, 
        description="Minimum story score to consider"
    )
    max_age_hours: float = Field(
        default=24.0, 
        ge=1.0, 
        le=168.0, 
        description="Maximum age of stories in hours"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO", 
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level"
    )
    json_logs: bool = Field(
        default=False, 
        description="Enable JSON structured logging"
    )
    
    # HN API Configuration
    hn_api_base_url: HttpUrl = Field(
        default="https://hacker-news.firebaseio.com/v0",
        description="Hacker News API base URL"
    )
    hn_request_delay: float = Field(
        default=0.1, 
        ge=0.0, 
        le=5.0, 
        description="Delay between HN API requests"
    )
    hn_max_concurrent: int = Field(
        default=10, 
        ge=1, 
        le=50, 
        description="Maximum concurrent HN API requests"
    )
    hn_timeout: float = Field(
        default=30.0, 
        ge=5.0, 
        le=120.0, 
        description="HN API request timeout"
    )
    
    # Runtime Configuration
    environment: str = Field(
        default="development", 
        pattern=r"^(development|staging|production)$",
        description="Runtime environment"
    )
    debug: bool = Field(
        default=False, 
        description="Enable debug mode"
    )
    dry_run: bool = Field(
        default=False, 
        description="Run without sending notifications"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        """Validate OpenAI API key format."""
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        if len(v) < 20:
            raise ValueError("OpenAI API key appears to be too short")
        return v
    
    @field_validator("slack_webhook_url")
    @classmethod
    def validate_slack_webhook(cls, v: HttpUrl) -> HttpUrl:
        """Validate Slack webhook URL."""
        if "hooks.slack.com" not in str(v):
            raise ValueError("Must be a valid Slack webhook URL")
        return v
    
    @field_validator("openai_model")
    @classmethod
    def validate_openai_model(cls, v: str) -> str:
        """Validate OpenAI model name."""
        valid_models = {
            "gpt-5-nano", "gpt-5-mini", "gpt-5"
        }
        if v not in valid_models:
            raise ValueError(f"Model must be one of: {', '.join(valid_models)}")
        return v
    
    
    @field_validator("slack_icon_emoji")
    @classmethod
    def validate_slack_emoji(cls, v: str) -> str:
        """Validate Slack emoji format."""
        if v and not (v.startswith(":") and v.endswith(":")):
            raise ValueError("Slack emoji must be in format :emoji:")
        return v

    def get_effective_topics(self) -> List[str]:
        """Get the effective list of topics based on configuration.
        
        Returns:
            Combined list of topics from direct topics, categories, and defaults
        """
        # Parse topics from string
        topics_list = [topic.strip() for topic in self.topics.split(",") if topic.strip()]
        effective_topics = set(topics_list)
        
        # Add topics from enabled categories
        if self.topic_categories:
            categories_list = [cat.strip() for cat in self.topic_categories.split(",") if cat.strip()]
            for category in categories_list:
                if category in TOPIC_CATEGORIES:
                    effective_topics.update(TOPIC_CATEGORIES[category])
        
        # Add all default topics if enabled
        if self.enable_all_default_topics:
            effective_topics.update(DEFAULT_TOPICS)
        
        return sorted(list(effective_topics))
    
    def get_topic_keywords(self) -> Set[str]:
        """Get all topic keywords for filtering (lowercased for matching)."""
        topics = self.get_effective_topics()
        return {topic.lower() for topic in topics}
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def get_log_config(self) -> dict:
        """Get logging configuration dictionary."""
        return {
            "level": self.log_level,
            "json": self.json_logs,
            "debug": self.debug
        }
    
    def validate_configuration(self) -> None:
        """Perform additional configuration validation.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate story limits
        if self.summary_max_stories > self.max_stories:
            raise ValueError(
                "summary_max_stories cannot be greater than max_stories"
            )
        
        # Validate topics
        if not self.get_effective_topics():
            raise ValueError("At least one topic must be configured")
        
        # Environment-specific validations
        if self.is_production():
            if self.debug:
                raise ValueError("Debug mode cannot be enabled in production")
            if self.dry_run:
                raise ValueError("Dry run mode cannot be enabled in production")
    
    def __str__(self) -> str:
        """String representation (hiding sensitive data)."""
        return (
            f"HNAgentSettings("
            f"env={self.environment}, "
            f"topics={len(self.get_effective_topics())}, "
            f"max_stories={self.max_stories})"
        )


@lru_cache()
def get_settings() -> HNAgentSettings:
    """Get cached application settings.
    
    Returns:
        Singleton HNAgentSettings instance
        
    Note:
        Uses lru_cache for singleton behavior. Clear cache if you need
        to reload settings during runtime.
    """
    settings = HNAgentSettings()
    settings.validate_configuration()
    return settings


def get_config() -> HNAgentSettings:
    """Alias for get_settings() for backward compatibility."""
    return get_settings()


def reload_settings() -> HNAgentSettings:
    """Force reload settings by clearing cache.
    
    Returns:
        Fresh HNAgentSettings instance
    """
    get_settings.cache_clear()
    return get_settings()