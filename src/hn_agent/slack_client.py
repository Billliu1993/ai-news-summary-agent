"""Slack webhook integration for sending story summaries."""

import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from urllib.parse import urlparse

import httpx
import structlog

from .models import Story, Summary, SlackMessage
from .config import HNAgentSettings

logger = structlog.get_logger(__name__)


# Slack message limits
SLACK_MAX_TEXT_LENGTH = 3000
SLACK_MAX_BLOCKS = 50
SLACK_MAX_SECTION_TEXT = 3000
SLACK_MAX_STORIES_PER_MESSAGE = 25


class SlackError(Exception):
    """Base exception for Slack-related errors."""

    pass


class WebhookError(SlackError):
    """Webhook request failed."""

    pass


class MessageTooLongError(SlackError):
    """Message exceeds Slack limits."""

    pass


class SlackClient:
    """Client for sending rich formatted messages to Slack via webhooks."""

    def __init__(self, settings: HNAgentSettings):
        """Initialize the Slack client.

        Args:
            settings: Application settings with Slack configuration
        """
        self.settings = settings
        self.webhook_url = str(settings.slack_webhook_url)
        self.username = settings.slack_username
        self.icon_emoji = settings.slack_icon_emoji
        self.timeout = settings.slack_timeout

        # HTTP client (initialized lazily)
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            "SlackClient initialized",
            webhook_domain=urlparse(self.webhook_url).netloc,
            username=self.username,
            timeout=self.timeout,
        )

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "HN-Summary-Agent/1.0",
                },
            )
        return self._client

    async def send_summary(self, summary: Summary, stories: List[Story]) -> bool:
        """Send a formatted summary to Slack with rich blocks.

        Args:
            summary: Summary object with content and metadata
            stories: List of Story objects that were summarized

        Returns:
            True if sent successfully, False otherwise
        """
        logger.info(
            "Preparing summary for Slack",
            story_count=len(stories),
            topics=summary.topics,
            summary_length=len(summary.content),
        )

        try:
            # Build rich message blocks
            blocks = self._build_summary_blocks(summary, stories)

            # Check message size and truncate if needed
            blocks = self._ensure_message_size_limits(blocks, summary.content)

            # Create fallback text
            fallback_text = self._create_fallback_text(summary, stories)

            # Send message
            success = await self._send_blocks_message(fallback_text, blocks)

            if success:
                logger.info("Summary sent to Slack successfully")
            else:
                logger.error("Failed to send summary to Slack")

            return success

        except Exception as e:
            logger.error("Error preparing summary for Slack", error=str(e))

            # Fallback: send simple text message
            return await self._send_fallback_summary(summary, stories)

    async def send_error_notification(
        self, error: str, context: Dict[str, Any], severity: str = "error"
    ) -> bool:
        """Send error notification to Slack with formatting.

        Args:
            error: Error message
            context: Additional context information
            severity: Error severity (error, warning, info)

        Returns:
            True if sent successfully
        """
        logger.info("Sending error notification to Slack", severity=severity)

        try:
            blocks = self._build_error_blocks(error, context, severity)
            fallback_text = f"{severity.upper()}: {error}"

            return await self._send_blocks_message(fallback_text, blocks)

        except Exception as e:
            logger.error("Failed to send error notification", error=str(e))
            return False

    async def _send_blocks_message(
        self, text: str, blocks: List[Dict[str, Any]]
    ) -> bool:
        """Send message with blocks to Slack webhook.

        Args:
            text: Fallback text
            blocks: Rich message blocks

        Returns:
            True if successful
        """
        payload = {
            "text": text,
            "blocks": blocks,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "unfurl_links": False,
            "unfurl_media": False,
        }

        return await self._send_webhook(payload)

    async def _send_webhook(self, payload: Dict[str, Any]) -> bool:
        """Send payload to Slack webhook with error handling.

        Args:
            payload: Slack webhook payload

        Returns:
            True if successful

        Raises:
            WebhookError: If webhook request fails
        """
        try:
            if self.settings.dry_run:
                logger.info(
                    "DRY RUN: Would send to Slack",
                    payload_size=len(json.dumps(payload)),
                )
                return True

            response = await self.client.post(self.webhook_url, json=payload)

            if response.status_code == 200:
                logger.debug("Slack webhook successful")
                return True
            elif response.status_code == 429:
                # Rate limited
                retry_after = response.headers.get("Retry-After", "60")
                logger.warning(
                    "Slack rate limit hit",
                    status_code=response.status_code,
                    retry_after=retry_after,
                )
                raise WebhookError(f"Rate limited, retry after {retry_after}s")
            else:
                error_text = response.text
                logger.error(
                    "Slack webhook failed",
                    status_code=response.status_code,
                    response=error_text,
                )
                raise WebhookError(f"HTTP {response.status_code}: {error_text}")

        except httpx.TimeoutException:
            logger.error("Slack webhook timeout")
            raise WebhookError("Request timeout")
        except httpx.RequestError as e:
            logger.error("Slack webhook request error", error=str(e))
            raise WebhookError(f"Request failed: {e}")

    def _build_summary_blocks(
        self, summary: Summary, stories: List[Story]
    ) -> List[Dict[str, Any]]:
        """Build rich Slack blocks for summary message.

        Args:
            summary: Summary object
            stories: List of stories

        Returns:
            List of Slack block elements
        """
        blocks = []

        # Header block
        topic_emojis = self._get_topic_emojis(summary.topics)
        header_text = f"{' '.join(topic_emojis)} Hacker News Summary"

        blocks.append(
            {"type": "header", "text": {"type": "plain_text", "text": header_text}}
        )

        # Summary info context - first line
        context_line1 = [
            f"📊 {summary.story_count} stories",
            f"⏰ {summary.generated_at.strftime('%H:%M %Z')}",
        ]

        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": " • ".join(context_line1)}],
            }
        )

        # Topics on separate line
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"🏷️ {', '.join(summary.topics)}"}
                ],
            }
        )

        # Divider
        blocks.append({"type": "divider"})

        # Main summary content - parse into structured sections
        summary_sections = self._parse_summary_sections(summary.content)

        for section in summary_sections:
            # Add section blocks
            blocks.extend(section)

        # Footer with stats
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"🔥 Top score: {max(s.score for s in stories)} pts • 💬 Total comments: {sum(s.comment_count for s in stories)}",
                    }
                ],
            }
        )

        return blocks

    def _build_story_cards(self, stories: List[Story]) -> List[Dict[str, Any]]:
        """Build story cards with clickable links.

        Args:
            stories: List of stories to create cards for

        Returns:
            List of story card blocks
        """
        if not stories:
            return []

        blocks = []

        # Section header
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*📋 Top {len(stories)} Stories*"},
            }
        )

        for story in stories:
            # Story card
            story_text = f"*<{self._get_story_link(story)}|{story.title}>*"

            # Add metadata
            metadata_parts = [
                f"⬆️ {story.score} pts",
                f"💬 {story.comment_count}",
                f"👤 {story.by}",
                f"⏰ {story.age_hours:.1f}h",
            ]

            story_text += f"\n{' • '.join(metadata_parts)}"

            # Add domain for external links
            if story.url and story.has_url:
                domain = urlparse(story.url).netloc
                story_text += f" • 🌐 {domain}"

            block = {"type": "section", "text": {"type": "mrkdwn", "text": story_text}}

            # Add accessory button for HN comments
            if story.comment_count > 0:
                hn_url = f"https://news.ycombinator.com/item?id={story.id}"
                block["accessory"] = {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": f"💬 {story.comment_count}",
                        "emoji": True,
                    },
                    "url": hn_url,
                    "action_id": f"hn_comments_{story.id}",
                }

            blocks.append(block)

        return blocks

    def _build_error_blocks(
        self, error: str, context: Dict[str, Any], severity: str
    ) -> List[Dict[str, Any]]:
        """Build error notification blocks.

        Args:
            error: Error message
            context: Error context
            severity: Error severity level

        Returns:
            List of error blocks
        """
        # Emoji mapping for severity
        severity_emojis = {"error": "🚨", "warning": "⚠️", "info": "ℹ️"}

        emoji = severity_emojis.get(severity, "❓")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} HN Agent {severity.title()}",
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Error:* {error}"},
            },
        ]

        # Add context if provided
        if context:
            context_text = "\n".join(
                [
                    f"*{key}:* {value}"
                    for key, value in context.items()
                    if value is not None
                ]
            )

            if context_text:
                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Context:*\n{context_text}",
                        },
                    }
                )

        # Add timestamp
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}",
                    }
                ],
            }
        )

        return blocks

    def _get_topic_emojis(self, topics: List[str]) -> List[str]:
        """Get emojis for topics.

        Args:
            topics: List of topic strings

        Returns:
            List of relevant emojis
        """
        topic_emoji_map = {
            "ai": "🤖",
            "artificial intelligence": "🤖",
            "machine learning": "🤖",
            "programming": "💻",
            "coding": "💻",
            "software": "💻",
            "development": "💻",
            "startups": "🚀",
            "startup": "🚀",
            "funding": "💰",
            "vc": "💰",
            "security": "🔒",
            "privacy": "🔒",
            "crypto": "₿",
            "blockchain": "⛓️",
            "web": "🌐",
            "mobile": "📱",
            "cloud": "☁️",
            "database": "💾",
        }

        emojis = []
        topic_lower = [t.lower() for t in topics]

        for topic, emoji in topic_emoji_map.items():
            if any(topic in t for t in topic_lower):
                if emoji not in emojis:
                    emojis.append(emoji)

        return emojis[:3]  # Limit to 3 emojis

    def _get_story_link(self, story: Story) -> str:
        """Get the primary link for a story.

        Args:
            story: Story object

        Returns:
            URL to link to (external URL or HN comments)
        """
        if story.url and story.has_url:
            return story.url
        else:
            # Link to HN comments for text posts
            return f"https://news.ycombinator.com/item?id={story.id}"

    def _ensure_message_size_limits(
        self, blocks: List[Dict[str, Any]], summary_content: str
    ) -> List[Dict[str, Any]]:
        """Ensure message fits within Slack limits.

        Args:
            blocks: Original blocks
            summary_content: Summary content for fallback

        Returns:
            Truncated blocks if needed
        """
        if len(blocks) <= SLACK_MAX_BLOCKS:
            return blocks

        logger.warning("Message too long, truncating", block_count=len(blocks))

        # Keep header, context, divider, and first few content blocks
        essential_blocks = blocks[:3]  # Header, context, divider
        content_blocks = [b for b in blocks[3:] if b.get("type") == "section"]
        other_blocks = [b for b in blocks[3:] if b.get("type") != "section"]

        # Add as many content blocks as possible
        available_space = (
            SLACK_MAX_BLOCKS - len(essential_blocks) - 2
        )  # Reserve space for truncation notice
        truncated_blocks = essential_blocks + content_blocks[:available_space]

        # Add truncation notice
        truncated_blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "_⚠️ Message truncated due to length limits_",
                },
            }
        )

        return truncated_blocks

    def _parse_summary_sections(self, content: str) -> List[List[Dict[str, Any]]]:
        """Parse summary content into structured Slack blocks by sections.

        Args:
            content: Full summary content

        Returns:
            List of block lists, one per section
        """
        sections = []
        lines = content.split("\n")
        current_section = []
        current_section_title = ""
        current_section_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a section header (starts with emoji)
            if self._is_section_header(line):
                # Save previous section if exists
                if current_section_content:
                    section_blocks = self._build_section_blocks(
                        current_section_title, current_section_content
                    )
                    if section_blocks:
                        sections.append(section_blocks)

                # Start new section
                current_section_title = line
                current_section_content = []
            else:
                # Add to current section content
                current_section_content.append(line)

        # Add final section
        if current_section_content:
            section_blocks = self._build_section_blocks(
                current_section_title, current_section_content
            )
            if section_blocks:
                sections.append(section_blocks)

        return sections

    def _is_section_header(self, line: str) -> bool:
        """Check if a line is a section header (starts with emoji).

        Args:
            line: Line to check

        Returns:
            True if it's a section header
        """
        # Common section header patterns
        section_patterns = ["🤖", "💾", "🧭", "📊", "🔬", "💻", "🚀", "🔒", "⚡", "🌐"]
        return any(line.startswith(pattern) for pattern in section_patterns)

    def _build_section_blocks(
        self, title: str, content: List[str]
    ) -> List[Dict[str, Any]]:
        """Build blocks for a summary section.

        Args:
            title: Section title (with emoji)
            content: List of content lines

        Returns:
            List of Slack blocks for this section
        """
        if not content:
            return []

        blocks = []

        # Section header if we have a title
        if title:
            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*"}}
            )

        # Parse content into story items and other content
        story_blocks = self._parse_story_items(content)
        blocks.extend(story_blocks)

        # Add small divider between sections
        blocks.append({"type": "divider"})

        return blocks

    def _parse_story_items(self, content_lines: List[str]) -> List[Dict[str, Any]]:
        """Parse structured story content into individual story blocks.

        Each story follows this format:
        **Title:** [title and metadata]
        **What:** [description]
        **Why:** [importance]
        **Link:** [url or "HN Discussion"]

        Args:
            content_lines: Lines of content for a section

        Returns:
            List of blocks for stories and other content
        """
        blocks = []
        current_story = {}

        for line in content_lines:
            line = line.strip()
            if not line:
                continue

            # Check for structured story fields
            if line.startswith("**Title:**"):
                # Save previous story if complete
                if self._is_complete_story(current_story):
                    story_block = self._create_structured_story_block(current_story)
                    if story_block:
                        blocks.append(story_block)

                # Start new story
                current_story = {
                    "title": line[10:].strip(),  # Remove "**Title:**"
                    "what": "",
                    "why": "",
                    "link": "",
                }
            elif line.startswith("**What:**") and current_story:
                current_story["what"] = line[9:].strip()  # Remove "**What:**"
            elif line.startswith("**Why:**") and current_story:
                current_story["why"] = line[8:].strip()  # Remove "**Why:**"
            elif line.startswith("**Link:**") and current_story:
                current_story["link"] = line[9:].strip()  # Remove "**Link:**"
            elif line and not current_story:
                # Standalone content line when no story is active
                blocks.append(
                    {"type": "section", "text": {"type": "mrkdwn", "text": line}}
                )

        # Handle final story
        if self._is_complete_story(current_story):
            story_block = self._create_structured_story_block(current_story)
            if story_block:
                blocks.append(story_block)

        return blocks

    def _is_complete_story(self, story: Dict[str, str]) -> bool:
        """Check if a story has all required fields.

        Args:
            story: Story dictionary

        Returns:
            True if story has title and at least what/why fields
        """
        return story and story.get("title") and (story.get("what") or story.get("why"))

    def _create_structured_story_block(
        self, story: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Create a formatted Slack block for a structured story.

        Args:
            story: Dictionary with title, what, why, link fields

        Returns:
            Slack block for the story, or None if invalid
        """
        if not self._is_complete_story(story):
            return None

        # Build the story text with clean formatting
        story_parts = []

        # Title (make it bold and clean)
        title = story["title"]
        story_parts.append(f"*{title}*")

        # What section
        if story.get("what"):
            story_parts.append(f"📝 {story['what']}")

        # Why section
        if story.get("why"):
            story_parts.append(f"💡 {story['why']}")

        # Link section - should always be HN discussion link
        if story.get("link"):
            if story["link"].startswith("https://news.ycombinator.com/item?id="):
                # HN discussion link
                story_parts.append(f"💬 <{story['link']}|HN Discussion>")
            else:
                # Fallback for any other format
                story_parts.append(f"🔗 {story['link']}")

        # Join all parts with newlines
        full_text = "\n".join(story_parts)

        # Truncate if too long for a single block
        if len(full_text) > SLACK_MAX_SECTION_TEXT:
            full_text = full_text[: SLACK_MAX_SECTION_TEXT - 3] + "..."

        return {"type": "section", "text": {"type": "mrkdwn", "text": full_text}}

    def _create_story_block(self, story_lines: List[str]) -> Optional[Dict[str, Any]]:
        """Create a formatted block for a single story.

        Args:
            story_lines: Lines containing story information

        Returns:
            Slack block for the story, or None if invalid
        """
        if not story_lines:
            return None

        # Combine all lines into formatted text
        story_text = []

        for line in story_lines:
            if line.startswith("- "):
                # Main headline - make it bold and clean up formatting
                headline = line[2:].strip()  # Remove "- "
                story_text.append(f"*{headline}*")
            else:
                # Detail lines - keep as-is but clean up indentation
                cleaned_line = line.lstrip(" -").strip()
                if cleaned_line:
                    story_text.append(cleaned_line)

        if not story_text:
            return None

        # Join with newlines and ensure proper length
        full_text = "\n".join(story_text)

        # Truncate if too long for a single block
        if len(full_text) > SLACK_MAX_SECTION_TEXT:
            full_text = full_text[: SLACK_MAX_SECTION_TEXT - 3] + "..."

        return {"type": "section", "text": {"type": "mrkdwn", "text": full_text}}

    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks that fit Slack limits.

        Args:
            text: Text to chunk
            max_length: Maximum length per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]

        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length > max_length:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    # Single word too long, split it
                    chunks.append(word[: max_length - 3] + "...")
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _create_fallback_text(self, summary: Summary, stories: List[Story]) -> str:
        """Create plain text fallback for clients that don't support blocks.

        Args:
            summary: Summary object
            stories: List of stories

        Returns:
            Plain text summary
        """
        header = f"📰 Hacker News Summary - {', '.join(summary.topics)} ({summary.story_count} stories)"

        # Truncate summary if too long
        content = summary.content
        if len(content) > 1000:
            content = content[:1000] + "..."

        return f"{header}\n\n{content}"

    async def _send_fallback_summary(
        self, summary: Summary, stories: List[Story]
    ) -> bool:
        """Send simple fallback message when rich formatting fails.

        Args:
            summary: Summary object
            stories: List of stories

        Returns:
            True if successful
        """
        logger.info("Sending fallback text-only summary")

        try:
            text = self._create_fallback_text(summary, stories)
            payload = {
                "text": text,
                "username": self.username,
                "icon_emoji": self.icon_emoji,
            }

            return await self._send_webhook(payload)

        except Exception as e:
            logger.error("Failed to send fallback summary", error=str(e))
            return False

    async def close(self):
        """Close the HTTP client connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("SlackClient closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
