"""Tests for SlackClient."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.hn_agent.models import SlackMessage, Summary
from src.hn_agent.slack_client import (
    SLACK_MAX_BLOCKS,
    SLACK_MAX_SECTION_TEXT,
    SlackClient,
    WebhookError,
)
from tests.conftest import create_story_with_overrides


class TestSlackClient:
    """Test cases for SlackClient."""

    @pytest.fixture
    def client(self, mock_settings):
        """Create Slack client for testing."""
        return SlackClient(mock_settings)

    @pytest.fixture
    def mock_http_response(self):
        """Mock successful HTTP response."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.text = "ok"
        return response

    @pytest.fixture
    def mock_error_response(self):
        """Mock error HTTP response."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 400
        response.text = "invalid_payload"
        return response

    @pytest.fixture
    def mock_rate_limit_response(self):
        """Mock rate limit HTTP response."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 429
        response.headers = {"Retry-After": "60"}
        response.text = "rate_limited"
        return response

    def test_client_initialization(self, client, mock_settings):
        """Test client is initialized correctly."""
        assert client.settings == mock_settings
        assert client.webhook_url == str(mock_settings.slack_webhook_url)
        assert client.username == mock_settings.slack_username
        assert client.icon_emoji == mock_settings.slack_icon_emoji
        assert client.timeout == mock_settings.slack_timeout
        assert client._client is None  # Lazily initialized

    def test_client_property_lazy_initialization(self, client):
        """Test HTTP client is created lazily."""
        http_client = client.client
        assert http_client is not None
        assert isinstance(http_client, httpx.AsyncClient)

        # Second access returns the same client
        assert client.client is http_client

    @pytest.mark.asyncio
    async def test_send_summary_success(
        self, client, sample_summary, sample_stories, mock_http_response
    ):
        """Test successful summary sending."""
        # Disable dry_run to test actual HTTP call
        client.settings.dry_run = False

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = mock_http_response

            result = await client.send_summary(sample_summary, sample_stories)

            assert result is True
            mock_post.assert_called_once()

            # Check the payload structure
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert "text" in payload
            assert "blocks" in payload
            assert "username" in payload
            assert payload["username"] == client.username

    @pytest.mark.asyncio
    async def test_send_summary_dry_run(self, client, sample_summary, sample_stories):
        """Test summary sending in dry run mode."""
        client.settings.dry_run = True

        result = await client.send_summary(sample_summary, sample_stories)

        assert result is True

    @pytest.mark.asyncio
    async def test_send_summary_webhook_error(
        self, client, sample_summary, sample_stories, mock_error_response
    ):
        """Test summary sending with webhook error."""
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = mock_error_response

            result = await client.send_summary(sample_summary, sample_stories)

            assert result is False

    @pytest.mark.asyncio
    async def test_send_summary_falls_back_on_error(
        self, client, sample_summary, sample_stories
    ):
        """Test that send_summary falls back to simple text on error."""
        with (
            patch.object(client, "_build_summary_blocks") as mock_build_blocks,
            patch.object(client, "_send_fallback_summary") as mock_fallback,
        ):

            # Make block building fail
            mock_build_blocks.side_effect = Exception("Block building failed")
            mock_fallback.return_value = True

            result = await client.send_summary(sample_summary, sample_stories)

            assert result is True
            mock_fallback.assert_called_once_with(sample_summary, sample_stories)

    @pytest.mark.asyncio
    async def test_send_error_notification_success(self, client, mock_http_response):
        """Test successful error notification."""
        error_msg = "Test error occurred"
        context = {"component": "summarizer", "retry_count": 3}

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = mock_http_response

            result = await client.send_error_notification(error_msg, context, "error")

            assert result is True
            mock_post.assert_called_once()

            # Check payload structure
            payload = mock_post.call_args[1]["json"]
            assert error_msg in payload["text"]
            assert len(payload["blocks"]) > 0

    @pytest.mark.asyncio
    async def test_send_error_notification_failure(self, client):
        """Test error notification sending failure."""
        with patch.object(client, "_send_blocks_message") as mock_send:
            mock_send.side_effect = Exception("Send failed")

            result = await client.send_error_notification("Error", {}, "error")

            assert result is False

    def test_build_summary_blocks_structure(
        self, client, sample_summary, sample_stories
    ):
        """Test summary blocks are built correctly."""
        blocks = client._build_summary_blocks(sample_summary, sample_stories)

        # Should have header, context, divider, content, and footer
        assert len(blocks) >= 4

        # Check header block
        header_block = blocks[0]
        assert header_block["type"] == "header"
        assert "Hacker News Summary" in header_block["text"]["text"]

        # Check for divider
        divider_blocks = [b for b in blocks if b["type"] == "divider"]
        assert len(divider_blocks) >= 1

        # Check for context blocks
        context_blocks = [b for b in blocks if b["type"] == "context"]
        assert len(context_blocks) >= 2  # Info and topics

    def test_build_summary_blocks_respects_limits(
        self, client, sample_summary, sample_stories
    ):
        """Test that block building respects Slack limits."""
        # Create a summary with lots of content that would exceed limits
        large_content = "🤖 AI/ML\n\n" + "\n\n".join(
            [
                f"**Title:** Story {i}\n**What:** Description {i}\n**Why:** Reason {i}\n**Link:** https://news.ycombinator.com/item?id={i}"
                for i in range(100)  # Way more than would fit
            ]
        )

        large_summary = Summary(
            content=large_content, story_count=100, topics=["AI"], total_score=5000
        )

        blocks = client._build_summary_blocks(large_summary, sample_stories)

        # Should not exceed Slack limits
        assert len(blocks) <= SLACK_MAX_BLOCKS

    def test_get_topic_emojis(self, client):
        """Test topic emoji mapping."""
        # Test various topics
        ai_topics = ["AI", "artificial intelligence", "machine learning"]
        ai_emojis = client._get_topic_emojis(ai_topics)
        assert "🤖" in ai_emojis

        programming_topics = ["programming", "coding", "software"]
        prog_emojis = client._get_topic_emojis(programming_topics)
        assert "💻" in prog_emojis

        startup_topics = ["startups", "funding", "vc"]
        startup_emojis = client._get_topic_emojis(startup_topics)
        assert any(emoji in startup_emojis for emoji in ["🚀", "💰"])

    def test_get_story_link_external_url(self, client):
        """Test story link for external URL."""
        story = create_story_with_overrides(url="https://example.com/article", id=12345)

        link = client._get_story_link(story)
        assert link == "https://example.com/article"

    def test_get_story_link_text_post(self, client):
        """Test story link for text post (Ask HN/Show HN)."""
        story = create_story_with_overrides(
            url=None, id=12345, text="This is an Ask HN post"
        )

        link = client._get_story_link(story)
        assert link == "https://news.ycombinator.com/item?id=12345"

    def test_ensure_message_size_limits_under_limit(self, client):
        """Test message size checking when under limits."""
        # Create blocks under the limit
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"Block {i}"}}
            for i in range(10)
        ]

        result = client._ensure_message_size_limits(blocks, "summary content")

        assert len(result) == 10
        assert result == blocks

    def test_ensure_message_size_limits_over_limit(self, client):
        """Test message size checking when over limits."""
        # Create blocks over the limit
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"Block {i}"}}
            for i in range(SLACK_MAX_BLOCKS + 10)
        ]

        result = client._ensure_message_size_limits(blocks, "summary content")

        assert len(result) <= SLACK_MAX_BLOCKS
        # Should have truncation notice
        assert any("truncated" in str(block).lower() for block in result)

    def test_parse_summary_sections(self, client):
        """Test parsing of summary content into sections."""
        content = """🤖 AI/ML

**Title:** AI Story 1
**What:** Description 1
**Why:** Reason 1
**Link:** https://news.ycombinator.com/item?id=1

**Title:** AI Story 2
**What:** Description 2
**Why:** Reason 2
**Link:** https://news.ycombinator.com/item?id=2

💻 Programming

**Title:** Programming Story
**What:** Programming description
**Why:** Programming reason
**Link:** https://news.ycombinator.com/item?id=3"""

        sections = client._parse_summary_sections(content)

        # Should have 2 sections (AI/ML and Programming)
        assert len(sections) >= 2

    def test_is_section_header(self, client):
        """Test section header detection."""
        assert client._is_section_header("🤖 AI/ML")
        assert client._is_section_header("💻 Programming")
        assert client._is_section_header("🚀 Startups")

        assert not client._is_section_header("**Title:** Regular title")
        assert not client._is_section_header("Regular text")

    def test_build_section_blocks(self, client):
        """Test building blocks for a section."""
        title = "🤖 AI/ML"
        content = [
            "**Title:** Test Story",
            "**What:** Test description",
            "**Why:** Test reason",
            "**Link:** https://news.ycombinator.com/item?id=12345",
        ]

        blocks = client._build_section_blocks(title, content)

        assert len(blocks) >= 2  # At least section header and content

        # First block should be section title
        assert blocks[0]["type"] == "section"
        assert title in blocks[0]["text"]["text"]

        # Should end with divider
        assert blocks[-1]["type"] == "divider"

    def test_parse_story_items(self, client):
        """Test parsing of story items from content."""
        content_lines = [
            "**Title:** Story 1",
            "**What:** Description 1",
            "**Why:** Reason 1",
            "**Link:** https://news.ycombinator.com/item?id=1",
            "",
            "**Title:** Story 2",
            "**What:** Description 2",
            "**Why:** Reason 2",
            "**Link:** https://news.ycombinator.com/item?id=2",
        ]

        blocks = client._parse_story_items(content_lines)

        # Should create blocks for both stories
        assert len(blocks) == 2
        assert all(block["type"] == "section" for block in blocks)

    def test_is_complete_story(self, client):
        """Test story completeness checking."""
        complete_story = {
            "title": "Test Story",
            "what": "Test description",
            "why": "Test reason",
            "link": "https://example.com",
        }
        assert client._is_complete_story(complete_story)

        incomplete_story = {
            "title": "Test Story",
            "what": "",  # Missing what
            "why": "",  # Missing why
            "link": "",
        }
        assert not client._is_complete_story(incomplete_story)

    def test_create_structured_story_block(self, client):
        """Test creation of structured story blocks."""
        story = {
            "title": "Test AI Story",
            "what": "This story is about AI advancement.",
            "why": "It shows progress in artificial intelligence.",
            "link": "https://news.ycombinator.com/item?id=12345",
        }

        block = client._create_structured_story_block(story)

        assert block is not None
        assert block["type"] == "section"
        text = block["text"]["text"]

        assert "Test AI Story" in text
        assert "This story is about AI advancement" in text
        assert "It shows progress in artificial intelligence" in text
        assert "HN Discussion" in text

    def test_create_structured_story_block_long_text(self, client):
        """Test story block creation with text that exceeds limits."""
        long_text = "Very long description. " * 500  # Way over limit

        story = {
            "title": "Test Story",
            "what": long_text,
            "why": "Test reason",
            "link": "https://news.ycombinator.com/item?id=12345",
        }

        block = client._create_structured_story_block(story)

        assert block is not None
        text = block["text"]["text"]
        assert len(text) <= SLACK_MAX_SECTION_TEXT
        assert text.endswith("...")

    def test_chunk_text_under_limit(self, client):
        """Test text chunking when text is under limit."""
        text = "Short text that fits in one chunk"

        chunks = client._chunk_text(text, 1000)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_over_limit(self, client):
        """Test text chunking when text exceeds limit."""
        words = ["word"] * 100
        text = " ".join(words)

        chunks = client._chunk_text(text, 50)  # Small limit

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_create_fallback_text(self, client, sample_summary, sample_stories):
        """Test creation of fallback text."""
        result = client._create_fallback_text(sample_summary, sample_stories)

        assert "Hacker News Summary" in result
        assert ", ".join(sample_summary.topics) in result
        assert str(sample_summary.story_count) in result
        assert sample_summary.content[:500] in result  # May be truncated

    def test_build_error_blocks(self, client):
        """Test error notification block building."""
        error = "Database connection failed"
        context = {
            "component": "hn_client",
            "retry_count": 3,
            "timestamp": "2024-01-01T10:00:00Z",
        }
        severity = "error"

        blocks = client._build_error_blocks(error, context, severity)

        # Should have header, error, context, and timestamp blocks
        assert len(blocks) >= 3

        # Check header
        header_block = blocks[0]
        assert header_block["type"] == "header"
        assert "🚨" in header_block["text"]["text"]
        assert "Error" in header_block["text"]["text"]

        # Check error block
        error_block = blocks[1]
        assert error_block["type"] == "section"
        assert error in error_block["text"]["text"]

    @pytest.mark.asyncio
    async def test_send_webhook_success(self, client, mock_http_response):
        """Test successful webhook sending."""
        client.settings.dry_run = False
        payload = {"text": "test message", "blocks": []}

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = mock_http_response

            result = await client._send_webhook(payload)

            assert result is True
            mock_post.assert_called_once_with(client.webhook_url, json=payload)

    @pytest.mark.asyncio
    async def test_send_webhook_rate_limited(self, client, mock_rate_limit_response):
        """Test webhook sending with rate limit."""
        client.settings.dry_run = False
        payload = {"text": "test message"}

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = mock_rate_limit_response

            with pytest.raises(WebhookError, match="Rate limited"):
                await client._send_webhook(payload)

    @pytest.mark.asyncio
    async def test_send_webhook_timeout(self, client):
        """Test webhook sending with timeout."""
        client.settings.dry_run = False
        payload = {"text": "test message"}

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(WebhookError, match="Request timeout"):
                await client._send_webhook(payload)

    @pytest.mark.asyncio
    async def test_send_webhook_network_error(self, client):
        """Test webhook sending with network error."""
        client.settings.dry_run = False
        payload = {"text": "test message"}

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(WebhookError, match="Request failed"):
                await client._send_webhook(payload)

    @pytest.mark.asyncio
    async def test_send_fallback_summary_success(
        self, client, sample_summary, sample_stories, mock_http_response
    ):
        """Test successful fallback summary sending."""
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = mock_http_response

            result = await client._send_fallback_summary(sample_summary, sample_stories)

            assert result is True

    @pytest.mark.asyncio
    async def test_send_fallback_summary_failure(
        self, client, sample_summary, sample_stories
    ):
        """Test fallback summary sending failure."""
        with patch.object(client, "_send_webhook") as mock_send:
            mock_send.side_effect = WebhookError("Webhook failed")

            result = await client._send_fallback_summary(sample_summary, sample_stories)

            assert result is False

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test client cleanup."""
        # Initialize the client
        _ = client.client
        assert client._client is not None

        await client.close()

        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_settings):
        """Test client as async context manager."""
        async with SlackClient(mock_settings) as client:
            assert isinstance(client, SlackClient)

        # Client should be closed after context
        assert client._client is None

    def test_slack_message_model_validation(self):
        """Test SlackMessage model validation."""
        # Valid message
        valid_msg = SlackMessage(
            text="Test message", username="TestBot", icon_emoji=":robot_face:"
        )
        assert valid_msg.text == "Test message"

        # Test payload conversion
        payload = valid_msg.to_payload()
        assert payload["text"] == "Test message"
        assert payload["username"] == "TestBot"
        assert payload["icon_emoji"] == ":robot_face:"

    def test_slack_message_model_validation_errors(self):
        """Test SlackMessage model validation errors."""
        # Empty text should raise error
        with pytest.raises(ValueError, match="Message text cannot be empty"):
            SlackMessage(text="")

    @pytest.mark.asyncio
    async def test_send_summary_with_long_content(self, client, sample_stories):
        """Test sending summary with content that exceeds Slack limits."""
        # Create a very long summary
        long_content = "Very long summary content. " * 1000
        long_summary = Summary(
            content=long_content,
            story_count=len(sample_stories),
            topics=["AI"],
            total_score=500,
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            result = await client.send_summary(long_summary, sample_stories)

            assert result is True
            # Should handle the long content gracefully

    @pytest.mark.asyncio
    async def test_send_blocks_message(self, client, mock_http_response):
        """Test sending message with blocks."""
        text = "Fallback text"
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Test block"}}]

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = mock_http_response

            result = await client._send_blocks_message(text, blocks)

            assert result is True

            # Check payload structure
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["text"] == text
            assert payload["blocks"] == blocks
            assert "unfurl_links" in payload
            assert payload["unfurl_links"] is False

    def test_http_client_headers(self, client):
        """Test that HTTP client has correct headers."""
        http_client = client.client

        assert "Content-Type" in http_client.headers
        assert http_client.headers["Content-Type"] == "application/json"
        assert "User-Agent" in http_client.headers
        assert http_client.headers["User-Agent"] == "HN-Summary-Agent/1.0"

    def test_client_timeout_configuration(self, client, mock_settings):
        """Test that client timeout is configured correctly."""
        http_client = client.client
        assert http_client.timeout.read == mock_settings.slack_timeout
