"""Hacker News AI Summary Agent."""

# Legacy import removed - agent.py was unused
from .models import Story, StoryType, Summary, Comment, SlackMessage
from .config import get_config, get_settings, HNAgentSettings
from .main import HackerNewsDigest, DigestStats, main_digest, run_health_check
from .hn_client import HackerNewsClient
from .summarizer import StorySummarizer
from .slack_client import SlackClient

__version__ = "0.1.0"
__all__ = [
    # Main entry points
    "HackerNewsDigest",
    "DigestStats", 
    "main_digest",
    "run_health_check",
    
    # Core models
    "Story", 
    "StoryType", 
    "Summary", 
    "Comment",
    "SlackMessage",
    
    # Configuration
    "get_config", 
    "get_settings", 
    "HNAgentSettings",
    
    # Service clients
    "HackerNewsClient",
    "StorySummarizer",
    "SlackClient",
    
# Removed HackerNewsAgent - was unused placeholder
]

# Package-level convenience functions
async def generate_digest(**kwargs):
    """Generate a Hacker News digest with optional configuration overrides.
    
    Args:
        **kwargs: Configuration overrides (topics, max_stories, etc.)
        
    Returns:
        Tuple of (Summary or None, DigestStats)
        
    Example:
        >>> import asyncio
        >>> from hn_agent import generate_digest
        >>> summary, stats = asyncio.run(generate_digest(topics=["AI", "startups"]))
    """
    # Apply any configuration overrides
    import os
    original_env = {}
    
    try:
        # Map kwargs to environment variables
        env_mapping = {
            "topics": "TOPICS",
            "max_stories": "MAX_STORIES", 
            "summary_max_stories": "SUMMARY_MAX_STORIES",
            "dry_run": "DRY_RUN",
            "debug": "DEBUG",
            "log_level": "LOG_LEVEL"
        }
        
        for key, value in kwargs.items():
            if key in env_mapping:
                env_var = env_mapping[key]
                original_env[env_var] = os.environ.get(env_var)
                
                if isinstance(value, list):
                    os.environ[env_var] = ",".join(str(v) for v in value)
                elif isinstance(value, bool):
                    os.environ[env_var] = "true" if value else "false"
                else:
                    os.environ[env_var] = str(value)
        
        # Reload settings to pick up changes
        from .config import reload_settings
        reload_settings()
        
        # Generate digest
        settings = get_settings()
        async with HackerNewsDigest(settings) as digest:
            return await digest.generate_digest()
            
    finally:
        # Restore original environment
        for env_var, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = original_value
        
        # Reload settings again to restore original state
        from .config import reload_settings
        reload_settings()


async def health_check():
    """Perform a health check on all components.
    
    Returns:
        Dictionary with health status information
        
    Example:
        >>> import asyncio
        >>> from hn_agent import health_check
        >>> health = asyncio.run(health_check())
        >>> print(health["overall"])
    """
    settings = get_settings()
    async with HackerNewsDigest(settings) as digest:
        return await digest.health_check()