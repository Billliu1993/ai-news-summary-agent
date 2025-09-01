"""Shared pytest fixtures for HN Agent tests."""

import asyncio
import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import HttpUrl

from src.hn_agent.config import HNAgentSettings
from src.hn_agent.hn_client import HackerNewsClient
from src.hn_agent.models import Comment, Story, StoryType, Summary
from src.hn_agent.slack_client import SlackClient
from src.hn_agent.summarizer import StorySummarizer


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings() -> HNAgentSettings:
    """Mock settings for testing."""
    return HNAgentSettings(
        openai_api_key="sk-test-key-12345678901234567890123456789012",
        slack_webhook_url=HttpUrl(
            "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        ),
        topics="AI,programming,startups",
        max_stories=50,
        summary_max_stories=15,
        dry_run=True,
        environment="development",
    )


@pytest.fixture
def sample_story_data() -> dict[str, Any]:
    """Sample story data from HN API."""
    return {
        "by": "testuser",
        "descendants": 42,
        "id": 38123456,
        "kids": [38123457, 38123458, 38123459],
        "score": 156,
        "text": None,
        "time": 1704067200,  # 2024-01-01 00:00:00 UTC
        "title": "New AI model achieves breakthrough in language understanding",
        "type": "story",
        "url": "https://example.com/ai-breakthrough",
        "dead": False,
        "deleted": False,
    }


@pytest.fixture
def sample_ask_hn_data() -> dict[str, Any]:
    """Sample Ask HN story data."""
    return {
        "by": "askuser",
        "descendants": 23,
        "id": 38123460,
        "kids": [38123461, 38123462],
        "score": 89,
        "text": "I'm building an AI-powered tool for developers. What are the key features you'd want?",
        "time": 1704070800,  # 2024-01-01 01:00:00 UTC
        "title": "Ask HN: What AI developer tools do you actually want?",
        "type": "story",
        "url": None,
        "dead": False,
        "deleted": False,
    }


@pytest.fixture
def sample_comment_data() -> dict[str, Any]:
    """Sample comment data from HN API."""
    return {
        "by": "commenter",
        "id": 38123457,
        "kids": [38123470, 38123471],
        "parent": 38123456,
        "text": "This is fascinating! The implications for natural language processing are huge.",
        "time": 1704067800,  # 2024-01-01 00:10:00 UTC
        "type": "comment",
        "dead": False,
        "deleted": False,
    }


@pytest.fixture
def sample_story(sample_story_data) -> Story:
    """Sample Story model instance."""
    return Story(**sample_story_data)


@pytest.fixture
def sample_ask_hn(sample_ask_hn_data) -> Story:
    """Sample Ask HN Story model instance."""
    return Story(**sample_ask_hn_data)


@pytest.fixture
def sample_comment(sample_comment_data) -> Comment:
    """Sample Comment model instance."""
    return Comment(**sample_comment_data)


@pytest.fixture
def sample_stories(sample_story, sample_ask_hn) -> list[Story]:
    """List of sample stories for testing."""
    # Create additional stories for testing
    story2 = Story(
        by="user2",
        descendants=15,
        id=38123465,
        kids=[38123466],
        score=75,
        time=1704071400,  # 2024-01-01 01:10:00 UTC
        title="Startup raises $50M for AI infrastructure platform",
        type=StoryType.STORY,
        url="https://example.com/startup-funding",
        dead=False,
        deleted=False,
    )

    story3 = Story(
        by="user3",
        descendants=31,
        id=38123470,
        kids=[38123471, 38123472, 38123473],
        score=120,
        time=1704064800,  # 2023-12-31 23:20:00 UTC
        title="Show HN: Open source programming language for AI",
        type=StoryType.SHOW,
        url="https://github.com/example/ai-lang",
        dead=False,
        deleted=False,
    )

    return [sample_story, sample_ask_hn, story2, story3]


@pytest.fixture
def sample_summary(sample_stories) -> Summary:
    """Sample Summary model instance."""
    return Summary(
        content="""🤖 AI/ML

**Title:** New AI model achieves breakthrough in language understanding
**What:** Researchers demonstrate significant improvements in natural language processing capabilities.
**Why:** Could enable more sophisticated AI applications across multiple industries.
**Link:** https://news.ycombinator.com/item?id=38123456

**Title:** Ask HN: What AI developer tools do you actually want?
**What:** Developer asking community about desired AI-powered development tools.
**Why:** Shows growing demand for AI integration in software development workflows.
**Link:** https://news.ycombinator.com/item?id=38123460

🚀 Startups

**Title:** Startup raises $50M for AI infrastructure platform
**What:** Company secures Series B funding to build AI infrastructure solutions.
**Why:** Indicates strong investor confidence in AI infrastructure market.
**Link:** https://news.ycombinator.com/item?id=38123465""",
        story_count=4,
        topics=["AI", "programming", "startups"],
        generated_at=datetime(2024, 1, 1, 12, 0, 0),
        total_score=440,
    )


@pytest.fixture
def mock_hn_api_responses() -> dict[str, Any]:
    """Mock responses for HN API endpoints."""
    return {
        "topstories": [38123456, 38123460, 38123465, 38123470, 38123475],
        "items": {
            38123456: {
                "by": "testuser",
                "descendants": 42,
                "id": 38123456,
                "kids": [38123457, 38123458, 38123459],
                "score": 156,
                "text": None,
                "time": 1704067200,
                "title": "New AI model achieves breakthrough in language understanding",
                "type": "story",
                "url": "https://example.com/ai-breakthrough",
            },
            38123460: {
                "by": "askuser",
                "descendants": 23,
                "id": 38123460,
                "kids": [38123461, 38123462],
                "score": 89,
                "text": "I'm building an AI-powered tool for developers. What are the key features you'd want?",
                "time": 1704070800,
                "title": "Ask HN: What AI developer tools do you actually want?",
                "type": "story",
                "url": None,
            },
            38123465: {
                "by": "user2",
                "descendants": 15,
                "id": 38123465,
                "kids": [38123466],
                "score": 75,
                "time": 1704071400,
                "title": "Startup raises $50M for AI infrastructure platform",
                "type": "story",
                "url": "https://example.com/startup-funding",
            },
            38123457: {
                "by": "commenter",
                "id": 38123457,
                "kids": [38123470, 38123471],
                "parent": 38123456,
                "text": "This is fascinating! The implications for natural language processing are huge.",
                "time": 1704067800,
                "type": "comment",
            },
        },
    }


@pytest.fixture
def mock_openai_response() -> dict[str, Any]:
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": """🤖 AI/ML

**Title:** New AI model achieves breakthrough in language understanding
**What:** Researchers demonstrate significant improvements in natural language processing capabilities.
**Why:** Could enable more sophisticated AI applications across multiple industries.
**Link:** https://news.ycombinator.com/item?id=38123456

**Title:** Ask HN: What AI developer tools do you actually want?
**What:** Developer asking community about desired AI-powered development tools.
**Why:** Shows growing demand for AI integration in software development workflows.
**Link:** https://news.ycombinator.com/item?id=38123460"""
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 150, "completion_tokens": 200, "total_tokens": 350},
    }


@pytest.fixture
def mock_slack_success_response() -> httpx.Response:
    """Mock successful Slack webhook response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.text = "ok"
    return response


@pytest.fixture
def mock_slack_error_response() -> httpx.Response:
    """Mock error Slack webhook response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 400
    response.text = "invalid_payload"
    return response


@pytest.fixture
async def mock_hn_client(mock_settings, mock_hn_api_responses) -> AsyncMock:
    """Mock HN client with predefined responses."""
    client = AsyncMock(spec=HackerNewsClient)

    # Mock topstories endpoint
    client.get_top_stories.return_value = mock_hn_api_responses["topstories"]

    # Mock individual item fetching
    async def mock_get_item(item_id: int):
        return mock_hn_api_responses["items"].get(item_id)

    async def mock_get_story(story_id: int):
        data = mock_hn_api_responses["items"].get(story_id)
        if data and data.get("type") == "story":
            return Story(**data)
        return None

    client.get_item.side_effect = mock_get_item
    client.get_story.side_effect = mock_get_story

    # Mock batch operations
    client.get_stories_batch.return_value = [
        Story(**mock_hn_api_responses["items"][38123456]),
        Story(**mock_hn_api_responses["items"][38123460]),
        Story(**mock_hn_api_responses["items"][38123465]),
    ]

    client.get_recent_stories.return_value = [
        Story(**mock_hn_api_responses["items"][38123456]),
        Story(**mock_hn_api_responses["items"][38123460]),
        Story(**mock_hn_api_responses["items"][38123465]),
    ]

    return client


@pytest.fixture
async def mock_summarizer(mock_settings, mock_openai_response) -> AsyncMock:
    """Mock summarizer with predefined responses."""
    summarizer = AsyncMock(spec=StorySummarizer)

    async def mock_summarize_stories(stories, topics):
        return mock_openai_response["choices"][0]["message"]["content"]

    summarizer.summarize_stories.side_effect = mock_summarize_stories
    return summarizer


@pytest.fixture
async def mock_slack_client(mock_settings) -> AsyncMock:
    """Mock Slack client with predefined responses."""
    client = AsyncMock(spec=SlackClient)
    client.send_summary.return_value = True
    client.send_error_notification.return_value = True
    return client


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Mock httpx AsyncClient for API calls."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.mark.asyncio
async def pytest_configure(config):
    """Configure pytest for async tests."""
    # Ensure asyncio mode is properly set
    config.option.asyncio_mode = "auto"


# Utility functions for tests
def create_mock_response(data: Any, status_code: int = 200) -> MagicMock:
    """Create a mock HTTP response."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = data
    response.text = json.dumps(data) if isinstance(data, dict) else str(data)
    return response


def create_story_with_overrides(**overrides) -> Story:
    """Create a story with specific field overrides."""
    base_data = {
        "by": "testuser",
        "descendants": 10,
        "id": 12345,
        "kids": [],
        "score": 50,
        "time": 1704067200,
        "title": "Test Story",
        "type": StoryType.STORY,
        "url": "https://example.com",
        "dead": False,
        "deleted": False,
    }
    base_data.update(overrides)
    return Story(**base_data)


def create_comment_with_overrides(**overrides) -> Comment:
    """Create a comment with specific field overrides."""
    base_data = {
        "by": "testuser",
        "id": 12346,
        "parent": 12345,
        "text": "Test comment",
        "time": 1704067200,
        "type": StoryType.COMMENT,
        "kids": [],
        "dead": False,
        "deleted": False,
    }
    base_data.update(overrides)
    return Comment(**base_data)
