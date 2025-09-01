"""Hacker News API client for fetching stories."""

import asyncio

import httpx
import structlog
from pydantic import ValidationError

from .models import Comment, Story, StoryType

logger = structlog.get_logger(__name__)


class HNAPIError(Exception):
    """Base exception for HN API errors."""

    pass


class HNRateLimitError(HNAPIError):
    """Rate limit exceeded."""

    pass


class HNNotFoundError(HNAPIError):
    """Item not found."""

    pass


class HackerNewsClient:
    """Async client for fetching stories and comments from Hacker News API."""

    def __init__(
        self,
        base_url: str = "https://hacker-news.firebaseio.com/v0",
        timeout: float = 30.0,
        max_concurrent_requests: int = 10,
        request_delay: float = 0.1,
    ):
        """Initialize the HN API client.

        Args:
            base_url: HN API base URL
            timeout: Request timeout in seconds
            max_concurrent_requests: Max concurrent requests via semaphore
            request_delay: Delay between requests for rate limiting
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.request_delay = request_delay

        # Rate limiting semaphore
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        # HTTP client (initialized lazily)
        self._client: httpx.AsyncClient | None = None

        logger.info("HN client initialized", base_url=base_url, timeout=timeout)

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": "HN-Summary-Agent/1.0"},
            )
        return self._client

    async def _make_request(self, endpoint: str) -> dict:
        """Make rate-limited request to HN API.

        Args:
            endpoint: API endpoint (without base URL)

        Returns:
            JSON response as dictionary

        Raises:
            HNAPIError: For API-related errors
            HNRateLimitError: When rate limited
            HNNotFoundError: When item not found
        """
        async with self._semaphore:
            # Rate limiting delay
            await asyncio.sleep(self.request_delay)

            url = f"{self.base_url}/{endpoint.lstrip('/')}"

            try:
                response = await self.client.get(url)

                if response.status_code == 404:
                    raise HNNotFoundError(f"Item not found: {endpoint}")
                elif response.status_code == 429:
                    raise HNRateLimitError("HN API rate limit exceeded")
                elif response.status_code != 200:
                    raise HNAPIError(f"HTTP {response.status_code}: {response.text}")

                data = response.json()
                if data is None:
                    raise HNNotFoundError(f"Item not found or deleted: {endpoint}")

                logger.debug("API request successful", endpoint=endpoint)
                return data

            except httpx.TimeoutException:
                raise HNAPIError(f"Request timeout for {endpoint}")
            except httpx.RequestError as e:
                raise HNAPIError(f"Request failed for {endpoint}: {e}")

    async def get_top_stories(self, limit: int = 30) -> list[int]:
        """Fetch top story IDs from HN API.

        Args:
            limit: Maximum number of story IDs to return

        Returns:
            List of story IDs

        Raises:
            HNAPIError: If API request fails
        """
        logger.info("Fetching top stories", limit=limit)

        try:
            data = await self._make_request("topstories.json")
            story_ids = data[:limit] if isinstance(data, list) else []

            logger.info("Retrieved top stories", count=len(story_ids))
            return story_ids

        except Exception as e:
            logger.error("Failed to fetch top stories", error=str(e))
            raise HNAPIError(f"Failed to fetch top stories: {e}") from e

    async def get_item(self, item_id: int) -> dict | None:
        """Fetch a single item (story/comment) by ID.

        Args:
            item_id: The HN item ID

        Returns:
            Item data dictionary or None if not found
        """
        try:
            data = await self._make_request(f"item/{item_id}.json")
            return data

        except HNNotFoundError:
            logger.debug("Item not found", item_id=item_id)
            return None
        except Exception as e:
            logger.error("Failed to fetch item", item_id=item_id, error=str(e))
            return None

    async def get_story(self, story_id: int) -> Story | None:
        """Fetch a single story by ID and parse into Story model.

        Args:
            story_id: The HN story ID

        Returns:
            Story object or None if not found/invalid
        """
        try:
            data = await self.get_item(story_id)
            if not data:
                return None

            # Parse into Story model
            story = Story(**data)

            # Validate it's actually a story type
            if story.type not in (StoryType.STORY, StoryType.ASK, StoryType.SHOW):
                logger.debug(
                    "Item is not a story type", item_id=story_id, item_type=story.type
                )
                return None

            logger.debug(
                "Story fetched successfully", story_id=story_id, title=story.title
            )
            return story

        except ValidationError as e:
            logger.warning("Invalid story data", story_id=story_id, error=str(e))
            return None
        except Exception as e:
            logger.error("Failed to fetch story", story_id=story_id, error=str(e))
            return None

    async def get_stories_batch(self, story_ids: list[int]) -> list[Story]:
        """Fetch multiple stories concurrently.

        Args:
            story_ids: List of story IDs to fetch

        Returns:
            List of valid Story objects (excludes None results)
        """
        logger.info("Fetching story batch", count=len(story_ids))

        # Fetch all stories concurrently
        stories_data = await asyncio.gather(
            *[self.get_story(story_id) for story_id in story_ids],
            return_exceptions=True,
        )

        # Filter out None results and exceptions
        valid_stories = []
        error_count = 0

        for i, story in enumerate(stories_data):
            if isinstance(story, Exception):
                logger.warning(
                    "Story fetch failed", story_id=story_ids[i], error=str(story)
                )
                error_count += 1
            elif story is not None:
                valid_stories.append(story)

        logger.info(
            "Story batch completed",
            requested=len(story_ids),
            successful=len(valid_stories),
            errors=error_count,
        )

        return valid_stories

    async def get_recent_stories(
        self, limit: int = 30, max_age_hours: float = 24.0
    ) -> list[Story]:
        """Fetch recent stories (within specified age).

        Args:
            limit: Maximum number of story IDs to fetch initially
            max_age_hours: Maximum age of stories in hours

        Returns:
            List of recent Story objects
        """
        logger.info("Fetching recent stories", limit=limit, max_age_hours=max_age_hours)

        # Get top story IDs
        story_ids = await self.get_top_stories(limit)

        # Fetch stories in batches
        all_stories = await self.get_stories_batch(story_ids)

        # Filter by age
        recent_stories = [
            story
            for story in all_stories
            if story.is_valid and story.age_hours <= max_age_hours
        ]

        logger.info(
            "Recent stories filtered",
            total_fetched=len(all_stories),
            recent_count=len(recent_stories),
        )

        return recent_stories

    async def get_comment(self, comment_id: int) -> Comment | None:
        """Fetch a single comment by ID.

        Args:
            comment_id: The HN comment ID

        Returns:
            Comment object or None if not found/invalid
        """
        try:
            data = await self.get_item(comment_id)
            if not data:
                return None

            # Parse into Comment model
            comment = Comment(**data)

            # Validate it's a comment
            if comment.type != StoryType.COMMENT:
                logger.debug(
                    "Item is not a comment", item_id=comment_id, item_type=comment.type
                )
                return None

            return comment

        except ValidationError as e:
            logger.warning("Invalid comment data", comment_id=comment_id, error=str(e))
            return None
        except Exception as e:
            logger.error("Failed to fetch comment", comment_id=comment_id, error=str(e))
            return None

    async def get_story_comments(
        self, story: Story, max_comments: int = 10
    ) -> list[Comment]:
        """Fetch top-level comments for a story.

        Args:
            story: Story object to get comments for
            max_comments: Maximum number of comments to fetch

        Returns:
            List of Comment objects
        """
        if not story.kids:
            return []

        comment_ids = story.kids[:max_comments]
        logger.info(
            "Fetching story comments", story_id=story.id, comment_count=len(comment_ids)
        )

        # Fetch comments concurrently
        comments_data = await asyncio.gather(
            *[self.get_comment(comment_id) for comment_id in comment_ids],
            return_exceptions=True,
        )

        # Filter valid comments
        valid_comments = [
            comment
            for comment in comments_data
            if isinstance(comment, Comment) and comment.is_valid and comment.has_text
        ]

        logger.info(
            "Story comments fetched",
            story_id=story.id,
            valid_comments=len(valid_comments),
        )

        return valid_comments

    async def close(self):
        """Close the HTTP client connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("HN client closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
