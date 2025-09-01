"""Tests for StorySummarizer."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from src.hn_agent.summarizer import (
    APIError,
    RateLimitError,
    StorySummarizer,
    SummarizerError,
    TokenLimitError,
)
from tests.conftest import create_story_with_overrides


class MockChatCompletion:
    """Mock ChatCompletion for testing."""

    def __init__(self, content: str):
        self.choices = [MagicMock()]
        self.choices[0].message.content = content


class MockGPT5Response:
    """Mock GPT-5 response for testing."""

    def __init__(self, content: str):
        self.output_text = content


class TestStorySummarizer:
    """Test cases for StorySummarizer."""

    @pytest.fixture
    def summarizer(self, mock_settings):
        """Create summarizer for testing."""
        return StorySummarizer(mock_settings)

    @pytest.fixture
    def sample_summary_content(self):
        """Sample summary content for testing."""
        return """🤖 AI/ML

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
**Link:** https://news.ycombinator.com/item?id=38123465"""

    def test_summarizer_initialization(self, mock_settings):
        """Test summarizer is initialized correctly."""
        summarizer = StorySummarizer(mock_settings)

        assert summarizer.settings == mock_settings
        assert summarizer.client is not None

    @pytest.mark.asyncio
    async def test_summarize_stories_success(
        self, summarizer, sample_stories, sample_summary_content
    ):
        """Test successful story summarization."""
        topics = ["AI", "programming", "startups"]

        with patch.object(summarizer, "_make_api_call_with_retry") as mock_api_call:
            mock_response = MockChatCompletion(sample_summary_content)
            mock_api_call.return_value = mock_response

            result = await summarizer.summarize_stories(sample_stories, topics)

            assert result == sample_summary_content
            mock_api_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_stories_empty_list(self, summarizer):
        """Test summarization with empty story list."""
        with pytest.raises(SummarizerError, match="No stories provided"):
            await summarizer.summarize_stories([], ["AI"])

    @pytest.mark.asyncio
    async def test_summarize_stories_limits_max_stories(
        self, summarizer, sample_summary_content
    ):
        """Test that summarization respects max_stories limit."""
        # Create more stories than the limit
        stories = [
            create_story_with_overrides(id=i, title=f"Story {i}") for i in range(20)
        ]
        topics = ["AI"]

        with patch.object(summarizer, "_make_api_call_with_retry") as mock_api_call:
            mock_response = MockChatCompletion(sample_summary_content)
            mock_api_call.return_value = mock_response

            await summarizer.summarize_stories(stories, topics, max_stories=5)

            # Check that the user prompt was built with only 5 stories
            args, kwargs = mock_api_call.call_args
            user_prompt = args[1]

            # Count occurrences of story titles in prompt
            story_count_in_prompt = user_prompt.count("Story ")
            assert story_count_in_prompt == 5

    @pytest.mark.asyncio
    async def test_make_api_call_with_retry_success_first_try(
        self, summarizer, sample_summary_content
    ):
        """Test API call succeeds on first attempt."""
        system_prompt = "Test system prompt"
        user_prompt = "Test user prompt"

        with patch.object(summarizer, "_make_api_call") as mock_api_call:
            mock_response = MockChatCompletion(sample_summary_content)
            mock_api_call.return_value = mock_response

            result = await summarizer._make_api_call_with_retry(
                system_prompt, user_prompt
            )

            assert result == mock_response
            mock_api_call.assert_called_once_with(system_prompt, user_prompt)

    @pytest.mark.asyncio
    async def test_make_api_call_with_retry_rate_limit_then_success(
        self, summarizer, sample_summary_content
    ):
        """Test API call succeeds after rate limit retry."""
        system_prompt = "Test system prompt"
        user_prompt = "Test user prompt"

        with (
            patch.object(summarizer, "_make_api_call") as mock_api_call,
            patch("asyncio.sleep") as mock_sleep,
        ):

            # First call fails with rate limit, second succeeds
            mock_response = MockChatCompletion(sample_summary_content)
            mock_api_call.side_effect = [
                RateLimitError("Rate limited", retry_after=2.0),
                mock_response,
            ]

            result = await summarizer._make_api_call_with_retry(
                system_prompt, user_prompt
            )

            assert result == mock_response
            assert mock_api_call.call_count == 2
            mock_sleep.assert_called_once_with(2.0)  # Should use retry_after

    @pytest.mark.asyncio
    async def test_make_api_call_with_retry_exponential_backoff(
        self, summarizer, sample_summary_content
    ):
        """Test exponential backoff for API errors."""
        system_prompt = "Test system prompt"
        user_prompt = "Test user prompt"

        with (
            patch.object(summarizer, "_make_api_call") as mock_api_call,
            patch("asyncio.sleep") as mock_sleep,
        ):

            mock_response = MockChatCompletion(sample_summary_content)
            mock_api_call.side_effect = [
                APIError("API Error 1"),
                APIError("API Error 2"),
                mock_response,
            ]

            result = await summarizer._make_api_call_with_retry(
                system_prompt, user_prompt
            )

            assert result == mock_response
            assert mock_api_call.call_count == 3
            assert mock_sleep.call_count == 2

            # Check exponential backoff delays
            delays = [call[0][0] for call in mock_sleep.call_args_list]
            assert delays[0] == 1.0  # First retry: base_delay * 2^0
            assert delays[1] == 2.0  # Second retry: base_delay * 2^1

    @pytest.mark.asyncio
    async def test_make_api_call_with_retry_max_retries_exceeded(self, summarizer):
        """Test that max retries limit is respected."""
        system_prompt = "Test system prompt"
        user_prompt = "Test user prompt"

        with (
            patch.object(summarizer, "_make_api_call") as mock_api_call,
            patch("asyncio.sleep"),
        ):

            mock_api_call.side_effect = APIError("Persistent API Error")

            with pytest.raises(SummarizerError, match="API error after .* attempts"):
                await summarizer._make_api_call_with_retry(system_prompt, user_prompt)

            # Should try max_retries + 1 times (initial + retries)
            expected_calls = summarizer.settings.openai_max_retries + 1
            assert mock_api_call.call_count == expected_calls

    @pytest.mark.asyncio
    async def test_make_api_call_gpt4_model(self, summarizer, sample_summary_content):
        """Test API call with GPT-4 model (legacy chat completions)."""
        # Set model to GPT-4
        summarizer.settings.openai_model = "gpt-4o"

        system_prompt = "Test system prompt"
        user_prompt = "Test user prompt"

        with patch.object(
            summarizer.client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_response = MockChatCompletion(sample_summary_content)
            mock_create.return_value = mock_response

            result = await summarizer._make_api_call(system_prompt, user_prompt)

            assert result == mock_response
            mock_create.assert_called_once()

            # Check call arguments
            call_args = mock_create.call_args[1]
            assert call_args["model"] == "gpt-4o"
            assert len(call_args["messages"]) == 2
            assert call_args["messages"][0]["role"] == "system"
            assert call_args["messages"][1]["role"] == "user"
            assert "temperature" in call_args

    @pytest.mark.asyncio
    async def test_make_api_call_gpt5_model(self, summarizer, sample_summary_content):
        """Test API call with GPT-5 model (new responses API)."""
        # Set model to GPT-5
        summarizer.settings.openai_model = "gpt-5-nano"

        system_prompt = "Test system prompt"
        user_prompt = "Test user prompt"

        with patch.object(summarizer.client.responses, "create") as mock_create:
            mock_response = MockGPT5Response(sample_summary_content)
            mock_create.return_value = mock_response

            result = await summarizer._make_api_call(system_prompt, user_prompt)

            assert result == mock_response
            mock_create.assert_called_once()

            # Check call arguments
            call_args = mock_create.call_args[1]
            assert call_args["model"] == "gpt-5-nano"
            assert "input" in call_args
            assert "reasoning" in call_args
            assert "text" in call_args

    @pytest.mark.asyncio
    async def test_make_api_call_rate_limit_error(self, summarizer):
        """Test handling of OpenAI rate limit errors."""
        system_prompt = "Test system prompt"
        user_prompt = "Test user prompt"

        with patch.object(summarizer.client.chat.completions, "create") as mock_create:
            # Mock OpenAI rate limit error
            mock_error = openai.RateLimitError(
                message="Rate limit exceeded", response=MagicMock(), body=None
            )
            mock_error.response.headers = {"retry-after": "30"}
            mock_create.side_effect = mock_error

            with pytest.raises(RateLimitError) as exc_info:
                await summarizer._make_api_call(system_prompt, user_prompt)

            assert exc_info.value.retry_after == 30.0

    @pytest.mark.asyncio
    async def test_make_api_call_token_limit_error(self, summarizer):
        """Test handling of token limit errors."""
        system_prompt = "Test system prompt"
        user_prompt = "Test user prompt"

        with patch.object(summarizer.client.chat.completions, "create") as mock_create:
            mock_error = openai.BadRequestError(
                message="maximum context length exceeded",
                response=MagicMock(),
                body=None,
            )
            mock_create.side_effect = mock_error

            with pytest.raises(TokenLimitError, match="Token limit exceeded"):
                await summarizer._make_api_call(system_prompt, user_prompt)

    @pytest.mark.asyncio
    async def test_make_api_call_general_api_error(self, summarizer):
        """Test handling of general OpenAI API errors."""
        system_prompt = "Test system prompt"
        user_prompt = "Test user prompt"

        with patch.object(summarizer.client.chat.completions, "create") as mock_create:
            mock_error = openai.APIError(
                message="Service unavailable", request=MagicMock(), body=None
            )
            mock_create.side_effect = mock_error

            with pytest.raises(APIError, match="OpenAI API error"):
                await summarizer._make_api_call(system_prompt, user_prompt)

    def test_build_system_prompt(self, summarizer):
        """Test system prompt building."""
        topics = ["AI", "machine learning", "startups"]

        result = summarizer._build_system_prompt(topics)

        assert "AI, machine learning, startups" in result
        assert "**Title:**" in result
        assert "**What:**" in result
        assert "**Why:**" in result
        assert "**Link:**" in result
        assert "🤖 AI/ML" in result

    def test_build_user_prompt(self, summarizer, sample_stories):
        """Test user prompt building."""
        topics = ["AI", "programming"]

        result = summarizer._build_user_prompt(sample_stories, topics)

        # Check that all stories are included
        for story in sample_stories:
            assert story.title in result
            assert f"HN ID: {story.id}" in result
            assert f"{story.score} points" in result

        # Check metadata is included
        assert "structured summary" in result.lower()
        assert "key reminders" in result.lower()

    def test_build_user_prompt_with_text_content(self, summarizer):
        """Test user prompt includes text content for Ask HN posts."""
        story = create_story_with_overrides(
            title="Ask HN: What's your favorite AI tool?",
            text="I'm curious about what AI tools people are actually using in their daily work. "
            * 10,
            url=None,
        )

        result = summarizer._build_user_prompt([story])

        assert story.text[:200] in result
        assert "..." in result  # Should be truncated

    def test_extract_summary_chat_completion(self, summarizer, sample_summary_content):
        """Test summary extraction from ChatCompletion response."""
        response = MockChatCompletion(sample_summary_content)

        result = summarizer._extract_summary(response)

        assert result == sample_summary_content

    def test_extract_summary_gpt5_response(self, summarizer, sample_summary_content):
        """Test summary extraction from GPT-5 response."""
        response = MockGPT5Response(sample_summary_content)

        result = summarizer._extract_summary(response)

        assert result == sample_summary_content

    def test_extract_summary_empty_content(self, summarizer):
        """Test handling of empty response content."""
        response = MockChatCompletion("")

        with pytest.raises(SummarizerError, match="No content"):
            summarizer._extract_summary(response)

    def test_extract_summary_no_choices(self, summarizer):
        """Test handling of response with no choices."""
        # Create a mock that doesn't have output_text to ensure it's treated as ChatCompletion
        response = MagicMock()
        response.choices = []
        # Explicitly remove output_text to avoid GPT-5 path
        del response.output_text

        with pytest.raises(SummarizerError, match="No response choices"):
            summarizer._extract_summary(response)

    def test_extract_summary_no_message(self, summarizer):
        """Test handling of response with no message."""
        # Create a mock that doesn't have output_text to ensure it's treated as ChatCompletion
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = None
        # Explicitly remove output_text to avoid GPT-5 path
        del response.output_text

        with pytest.raises(SummarizerError, match="No content"):
            summarizer._extract_summary(response)

    def test_clean_exclusion_messages_well_formed(self, summarizer):
        """Test cleaning preserves well-formed content."""
        content = """🤖 AI/ML

**Title:** Test Story
**What:** This is a test story about AI.
**Why:** It's important for testing purposes.
**Link:** https://news.ycombinator.com/item?id=12345

💻 Programming

**Title:** Another Test Story  
**What:** This is about programming.
**Why:** Programming is fundamental to software development.
**Link:** https://news.ycombinator.com/item?id=12346"""

        result = summarizer._clean_exclusion_messages(content)

        # Should preserve all well-formed stories
        assert "Test Story" in result
        assert "Another Test Story" in result
        assert "🤖 AI/ML" in result
        assert "💻 Programming" in result

    def test_clean_exclusion_messages_filters_malformed(self, summarizer):
        """Test cleaning filters out malformed content."""
        content = """🤖 AI/ML

**Title:** Well-formed Story
**What:** This story has all required fields.
**Why:** It should be preserved.
**Link:** https://news.ycombinator.com/item?id=12345

**Title:** Incomplete Story
**What:** This story is missing the Why and Link fields.

Some random text that doesn't follow the format.

💻 Programming

**Title:** Another Good Story
**What:** This is properly formatted.
**Why:** It has all required fields.
**Link:** https://news.ycombinator.com/item?id=12347"""

        result = summarizer._clean_exclusion_messages(content)

        # Should preserve well-formed stories
        assert "Well-formed Story" in result
        assert "Another Good Story" in result

        # Should filter out malformed content
        assert "Incomplete Story" not in result
        assert "Some random text" not in result

    def test_clean_exclusion_messages_preserves_section_headers(self, summarizer):
        """Test cleaning preserves section headers."""
        content = """🤖 AI/ML

**Title:** AI Story
**What:** About artificial intelligence.
**Why:** AI is important.
**Link:** https://news.ycombinator.com/item?id=12345

🚀 Startups

**Title:** Startup Story
**What:** About a new startup.
**Why:** Startups drive innovation.
**Link:** https://news.ycombinator.com/item?id=12346"""

        result = summarizer._clean_exclusion_messages(content)

        assert "🤖 AI/ML" in result
        assert "🚀 Startups" in result

    @pytest.mark.asyncio
    async def test_close(self, summarizer):
        """Test summarizer cleanup."""
        with patch.object(summarizer.client, "close") as mock_close:
            await summarizer.close()
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_settings):
        """Test summarizer as async context manager."""
        with patch("src.hn_agent.summarizer.AsyncOpenAI") as mock_openai:
            # Mock the client with async close method
            mock_client = AsyncMock()
            mock_client.close = AsyncMock()
            mock_openai.return_value = mock_client

            async with StorySummarizer(mock_settings) as summarizer:
                assert isinstance(summarizer, StorySummarizer)

            # Verify close was called
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_stories_api_call_failure(self, summarizer, sample_stories):
        """Test handling of API call failure during summarization."""
        topics = ["AI"]

        with patch.object(summarizer, "_make_api_call_with_retry") as mock_api_call:
            mock_api_call.side_effect = SummarizerError("API call failed")

            with pytest.raises(SummarizerError, match="API call failed"):
                await summarizer.summarize_stories(sample_stories, topics)

    def test_build_user_prompt_escapes_special_characters(self, summarizer):
        """Test that user prompt properly handles special characters in titles."""
        story = create_story_with_overrides(
            title="Story with \"quotes\" and 'apostrophes' & ampersands",
            text="Text with <html> tags and other special chars: @#$%",
            url=None,  # Make it a text post so text content is included
        )

        result = summarizer._build_user_prompt([story])

        # Should include the title and text as-is (let OpenAI handle escaping)
        assert story.title in result
        assert "Text with <html> tags" in result

    @pytest.mark.asyncio
    async def test_concurrent_summarization(
        self, mock_settings, sample_stories, sample_summary_content
    ):
        """Test concurrent summarization requests."""
        summarizer1 = StorySummarizer(mock_settings)
        summarizer2 = StorySummarizer(mock_settings)

        with (
            patch.object(summarizer1, "_make_api_call_with_retry") as mock_api_call1,
            patch.object(summarizer2, "_make_api_call_with_retry") as mock_api_call2,
        ):

            mock_response = MockChatCompletion(sample_summary_content)
            mock_api_call1.return_value = mock_response
            mock_api_call2.return_value = mock_response

            # Run concurrent summarizations
            tasks = [
                summarizer1.summarize_stories(sample_stories[:2], ["AI"]),
                summarizer2.summarize_stories(sample_stories[2:], ["programming"]),
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 2
            assert all(result == sample_summary_content for result in results)
            mock_api_call1.assert_called_once()
            mock_api_call2.assert_called_once()
