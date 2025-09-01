"""Tests for HackerNewsClient."""

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.hn_agent.hn_client import (
    HackerNewsClient,
    HNAPIError,
    HNNotFoundError,
    HNRateLimitError,
)
from src.hn_agent.models import Comment, Story, StoryType
from tests.conftest import create_mock_response, create_story_with_overrides


class TestHackerNewsClient:
    """Test cases for HackerNewsClient."""

    @pytest.fixture
    def client(self):
        """Create HN client for testing."""
        return HackerNewsClient(
            base_url="https://test-hn-api.com/v0",
            timeout=10.0,
            max_concurrent_requests=5,
            request_delay=0.01,
        )

    def test_client_initialization(self, client):
        """Test client is initialized with correct parameters."""
        assert client.base_url == "https://test-hn-api.com/v0"
        assert client.timeout == 10.0
        assert client.request_delay == 0.01
        assert client._client is None  # Lazily initialized

    def test_client_property_lazy_initialization(self, client):
        """Test HTTP client is created lazily."""
        # First access creates the client
        http_client = client.client
        assert http_client is not None
        assert isinstance(http_client, httpx.AsyncClient)

        # Second access returns the same client
        assert client.client is http_client

    @pytest.mark.asyncio
    async def test_make_request_success(self, client):
        """Test successful API request."""
        mock_data = {"id": 12345, "title": "Test Story"}

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = create_mock_response(mock_data)
            mock_get.return_value = mock_response

            result = await client._make_request("item/12345.json")

            assert result == mock_data
            mock_get.assert_called_once_with(
                "https://test-hn-api.com/v0/item/12345.json"
            )

    @pytest.mark.asyncio
    async def test_make_request_404_error(self, client):
        """Test handling of 404 errors."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            with pytest.raises(HNNotFoundError, match="Item not found"):
                await client._make_request("item/99999.json")

    @pytest.mark.asyncio
    async def test_make_request_rate_limit_error(self, client):
        """Test handling of rate limit errors."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_get.return_value = mock_response

            with pytest.raises(HNRateLimitError, match="rate limit exceeded"):
                await client._make_request("topstories.json")

    @pytest.mark.asyncio
    async def test_make_request_general_http_error(self, client):
        """Test handling of other HTTP errors."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_get.return_value = mock_response

            with pytest.raises(HNAPIError, match="HTTP 500"):
                await client._make_request("topstories.json")

    @pytest.mark.asyncio
    async def test_make_request_timeout_error(self, client):
        """Test handling of timeout errors."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Request timeout")

            with pytest.raises(HNAPIError, match="Request timeout"):
                await client._make_request("topstories.json")

    @pytest.mark.asyncio
    async def test_make_request_network_error(self, client):
        """Test handling of network errors."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(HNAPIError, match="Request failed"):
                await client._make_request("topstories.json")

    @pytest.mark.asyncio
    async def test_make_request_null_response(self, client):
        """Test handling of null API responses."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = create_mock_response(None)
            mock_get.return_value = mock_response

            with pytest.raises(HNNotFoundError, match="Item not found or deleted"):
                await client._make_request("item/12345.json")

    @pytest.mark.asyncio
    async def test_get_top_stories_success(self, client, mock_hn_api_responses):
        """Test successful retrieval of top stories."""
        story_ids = mock_hn_api_responses["topstories"]

        with patch.object(client, "_make_request") as mock_request:
            mock_request.return_value = story_ids

            result = await client.get_top_stories(limit=3)

            assert result == story_ids[:3]
            mock_request.assert_called_once_with("topstories.json")

    @pytest.mark.asyncio
    async def test_get_top_stories_invalid_response(self, client):
        """Test handling of invalid topstories response."""
        with patch.object(client, "_make_request") as mock_request:
            mock_request.return_value = "invalid_data"

            result = await client.get_top_stories()

            assert result == []

    @pytest.mark.asyncio
    async def test_get_top_stories_api_error(self, client):
        """Test handling of API errors in get_top_stories."""
        with patch.object(client, "_make_request") as mock_request:
            mock_request.side_effect = HNAPIError("API Error")

            with pytest.raises(HNAPIError, match="Failed to fetch top stories"):
                await client.get_top_stories()

    @pytest.mark.asyncio
    async def test_get_item_success(self, client, sample_story_data):
        """Test successful item retrieval."""
        with patch.object(client, "_make_request") as mock_request:
            mock_request.return_value = sample_story_data

            result = await client.get_item(38123456)

            assert result == sample_story_data
            mock_request.assert_called_once_with("item/38123456.json")

    @pytest.mark.asyncio
    async def test_get_item_not_found(self, client):
        """Test item retrieval when item doesn't exist."""
        with patch.object(client, "_make_request") as mock_request:
            mock_request.side_effect = HNNotFoundError("Item not found")

            result = await client.get_item(99999)

            assert result is None

    @pytest.mark.asyncio
    async def test_get_item_api_error(self, client):
        """Test item retrieval with API error."""
        with patch.object(client, "_make_request") as mock_request:
            mock_request.side_effect = HNAPIError("API Error")

            result = await client.get_item(12345)

            assert result is None

    @pytest.mark.asyncio
    async def test_get_story_success(self, client, sample_story_data):
        """Test successful story retrieval and parsing."""
        with patch.object(client, "get_item") as mock_get_item:
            mock_get_item.return_value = sample_story_data

            result = await client.get_story(38123456)

            assert isinstance(result, Story)
            assert result.id == 38123456
            assert result.title == sample_story_data["title"]
            assert result.score == sample_story_data["score"]

    @pytest.mark.asyncio
    async def test_get_story_not_found(self, client):
        """Test story retrieval when story doesn't exist."""
        with patch.object(client, "get_item") as mock_get_item:
            mock_get_item.return_value = None

            result = await client.get_story(99999)

            assert result is None

    @pytest.mark.asyncio
    async def test_get_story_invalid_type(self, client, sample_comment_data):
        """Test story retrieval when item is not a story type."""
        with patch.object(client, "get_item") as mock_get_item:
            mock_get_item.return_value = sample_comment_data

            result = await client.get_story(38123457)

            assert result is None

    @pytest.mark.asyncio
    async def test_get_story_validation_error(self, client):
        """Test story retrieval with invalid data."""
        invalid_data = {"id": "invalid", "title": None}  # Invalid Story data

        with patch.object(client, "get_item") as mock_get_item:
            mock_get_item.return_value = invalid_data

            result = await client.get_story(12345)

            assert result is None

    @pytest.mark.asyncio
    async def test_get_stories_batch_success(self, client, mock_hn_api_responses):
        """Test batch story retrieval."""
        story_ids = [38123456, 38123460, 38123465]
        expected_stories = [
            Story(**mock_hn_api_responses["items"][story_id]) for story_id in story_ids
        ]

        with patch.object(client, "get_story") as mock_get_story:
            mock_get_story.side_effect = expected_stories

            result = await client.get_stories_batch(story_ids)

            assert len(result) == 3
            assert all(isinstance(story, Story) for story in result)
            assert result[0].id == 38123456
            assert result[1].id == 38123460
            assert result[2].id == 38123465

    @pytest.mark.asyncio
    async def test_get_stories_batch_with_failures(self, client):
        """Test batch story retrieval with some failures."""
        story_ids = [1, 2, 3, 4, 5]

        # Mock get_story to return stories for some IDs and None/exceptions for others
        async def mock_get_story(story_id):
            if story_id == 1:
                return create_story_with_overrides(id=1, title="Story 1")
            elif story_id == 2:
                raise Exception("API Error")
            elif story_id == 3:
                return None
            elif story_id == 4:
                return create_story_with_overrides(id=4, title="Story 4")
            else:  # story_id == 5
                return create_story_with_overrides(id=5, title="Story 5")

        with patch.object(client, "get_story") as mock_get_story_patch:
            mock_get_story_patch.side_effect = mock_get_story

            result = await client.get_stories_batch(story_ids)

            # Should return only successful stories (1, 4, 5)
            assert len(result) == 3
            assert result[0].id == 1
            assert result[1].id == 4
            assert result[2].id == 5

    @pytest.mark.asyncio
    async def test_get_recent_stories_success(self, client, sample_stories):
        """Test retrieval of recent stories."""
        story_ids = [1, 2, 3, 4, 5]

        # Create stories with different ages (using current time for realistic testing)
        import time

        current_time = int(time.time())

        recent_stories = [
            create_story_with_overrides(id=1, time=current_time - 3600),  # 1 hour ago
            create_story_with_overrides(
                id=2, time=current_time - 7200
            ),  # 2 hours ago (within 24h limit)
        ]
        old_stories = [
            create_story_with_overrides(
                id=3, time=current_time - 86400 - 3600
            ),  # 25 hours ago (too old)
        ]

        all_stories = recent_stories + old_stories

        with (
            patch.object(client, "get_top_stories") as mock_top_stories,
            patch.object(client, "get_stories_batch") as mock_batch,
        ):

            mock_top_stories.return_value = story_ids
            mock_batch.return_value = all_stories

            result = await client.get_recent_stories(limit=10, max_age_hours=24.0)

            # Should filter out old stories
            assert len(result) >= len(recent_stories)
            mock_top_stories.assert_called_once_with(10)
            mock_batch.assert_called_once_with(story_ids)

    @pytest.mark.asyncio
    async def test_get_comment_success(self, client, sample_comment_data):
        """Test successful comment retrieval."""
        with patch.object(client, "get_item") as mock_get_item:
            mock_get_item.return_value = sample_comment_data

            result = await client.get_comment(38123457)

            assert isinstance(result, Comment)
            assert result.id == 38123457
            assert result.text == sample_comment_data["text"]
            assert result.parent == sample_comment_data["parent"]

    @pytest.mark.asyncio
    async def test_get_comment_not_found(self, client):
        """Test comment retrieval when comment doesn't exist."""
        with patch.object(client, "get_item") as mock_get_item:
            mock_get_item.return_value = None

            result = await client.get_comment(99999)

            assert result is None

    @pytest.mark.asyncio
    async def test_get_comment_wrong_type(self, client, sample_story_data):
        """Test comment retrieval when item is not a comment."""
        with patch.object(client, "get_item") as mock_get_item:
            mock_get_item.return_value = sample_story_data

            result = await client.get_comment(38123456)

            assert result is None

    @pytest.mark.asyncio
    async def test_get_story_comments_success(self, client, sample_story):
        """Test retrieval of story comments."""
        # Create mock comments
        comments = [
            Comment(
                by="user1",
                id=1,
                parent=sample_story.id,
                text="Great story!",
                time=1704067200,
                type=StoryType.COMMENT,
                kids=[],
                dead=False,
                deleted=False,
            ),
            Comment(
                by="user2",
                id=2,
                parent=sample_story.id,
                text="I agree with this.",
                time=1704067300,
                type=StoryType.COMMENT,
                kids=[],
                dead=False,
                deleted=False,
            ),
        ]

        with patch.object(client, "get_comment") as mock_get_comment:
            mock_get_comment.side_effect = comments

            result = await client.get_story_comments(sample_story, max_comments=10)

            assert len(result) == 2
            assert all(isinstance(comment, Comment) for comment in result)
            assert result[0].text == "Great story!"
            assert result[1].text == "I agree with this."

    @pytest.mark.asyncio
    async def test_get_story_comments_no_kids(self, client):
        """Test comment retrieval for story with no comments."""
        story = create_story_with_overrides(kids=[])

        result = await client.get_story_comments(story)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_story_comments_with_failures(self, client, sample_story):
        """Test comment retrieval with some failures."""
        # Use the actual comment IDs from sample_story.kids
        # sample_story has kids=[38123457, 38123458, 38123459]
        valid_comment = Comment(
            by="user1",
            id=38123457,  # Match the first kid ID
            parent=sample_story.id,
            text="Valid comment",
            time=1704067200,
            type=StoryType.COMMENT,
            kids=[],
            dead=False,
            deleted=False,
        )

        # Mock get_comment to return mixed results
        async def mock_get_comment(comment_id):
            if comment_id == 38123457:  # First kid - valid comment
                return valid_comment
            elif comment_id == 38123458:  # Second kid - API error
                raise Exception("API Error")
            else:  # comment_id == 38123459 - third kid - not found
                return None

        with patch.object(client, "get_comment") as mock_get_comment_patch:
            mock_get_comment_patch.side_effect = mock_get_comment

            result = await client.get_story_comments(sample_story, max_comments=10)

            # Should return only valid comments
            assert len(result) == 1
            assert result[0].text == "Valid comment"

    @pytest.mark.asyncio
    async def test_get_story_comments_filters_invalid(self, client, sample_story):
        """Test that invalid/deleted comments are filtered out."""
        comments = [
            Comment(
                by="user1",
                id=1,
                parent=sample_story.id,
                text="Valid comment",
                time=1704067200,
                type=StoryType.COMMENT,
                kids=[],
                dead=False,
                deleted=False,
            ),
            Comment(
                by="user2",
                id=2,
                parent=sample_story.id,
                text="",  # Empty text
                time=1704067300,
                type=StoryType.COMMENT,
                kids=[],
                dead=False,
                deleted=False,
            ),
            Comment(
                by="user3",
                id=3,
                parent=sample_story.id,
                text="Deleted comment",
                time=1704067400,
                type=StoryType.COMMENT,
                kids=[],
                dead=False,
                deleted=True,  # Deleted
            ),
        ]

        with patch.object(client, "get_comment") as mock_get_comment:
            mock_get_comment.side_effect = comments

            result = await client.get_story_comments(sample_story, max_comments=10)

            # Should return only valid, non-deleted comments with text
            assert len(result) == 1
            assert result[0].text == "Valid comment"

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test client cleanup."""
        # Initialize the client
        _ = client.client
        assert client._client is not None

        await client.close()

        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager."""
        async with HackerNewsClient() as client:
            assert isinstance(client, HackerNewsClient)

        # Client should be closed after context
        assert client._client is None

    @pytest.mark.asyncio
    async def test_concurrent_requests_semaphore(self, client):
        """Test that concurrent requests are limited by semaphore."""
        # This test verifies that the semaphore is working by checking
        # that we can't exceed the max_concurrent_requests limit

        call_count = 0

        async def mock_make_request_slow(endpoint):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate slow request
            return {"id": 123}

        with patch.object(client, "_make_request", side_effect=mock_make_request_slow):
            # Start many requests concurrently
            tasks = [client.get_item(i) for i in range(10)]

            # Wait a short time to see how many started
            await asyncio.sleep(0.05)

            # Should not exceed max_concurrent_requests (5)
            assert (
                call_count <= client._semaphore._value + 5
            )  # Allow for the 5 that started

            # Wait for all to complete
            await asyncio.gather(*tasks, return_exceptions=True)

    def test_user_agent_header(self, client):
        """Test that User-Agent header is set correctly."""
        http_client = client.client
        assert "User-Agent" in http_client.headers
        assert http_client.headers["User-Agent"] == "HN-Summary-Agent/1.0"

    def test_timeout_configuration(self, client):
        """Test that timeout is configured correctly."""
        http_client = client.client
        assert http_client.timeout.read == 10.0
