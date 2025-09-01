"""OpenAI-powered summarization service for Hacker News stories."""

import asyncio
import time
from typing import List, Optional

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from typing import Any
import structlog

from .models import Story
from .config import HNAgentSettings

logger = structlog.get_logger(__name__)





class SummarizerError(Exception):
    """Base exception for summarizer errors."""
    pass


class RateLimitError(SummarizerError):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class APIError(SummarizerError):
    """OpenAI API error."""
    pass


class TokenLimitError(SummarizerError):
    """Token limit exceeded error."""
    pass


class StorySummarizer:
    """Handles story summarization using OpenAI's API with comprehensive features."""
    
    def __init__(self, settings: HNAgentSettings):
        """Initialize the summarizer.
        
        Args:
            settings: Application settings with OpenAI configuration
        """
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        logger.info(
            "StorySummarizer initialized",
            model=settings.openai_model,
            max_tokens=settings.openai_max_tokens,
            temperature=settings.openai_temperature
        )
    
    async def summarize_stories(
        self, 
        stories: List[Story], 
        topics: List[str],
        max_stories: Optional[int] = None
    ) -> str:
        """Summarize a list of Hacker News stories.
        
        Args:
            stories: List of Story objects
            topics: List of topics to focus on
            max_stories: Maximum stories to include (uses config default if None)
            
        Returns:
            Summary text
            
        Raises:
            SummarizerError: If summarization fails after all retries
        """
        if not stories:
            raise SummarizerError("No stories provided for summarization")
        
        # Limit stories if needed
        max_count = max_stories or self.settings.summary_max_stories
        limited_stories = stories[:max_count]
        
        logger.info(
            "Starting story summarization",
            total_stories=len(stories),
            summarizing=len(limited_stories),
            topics=topics
        )
        
        # Build prompts
        system_prompt = self._build_system_prompt(topics)
        user_prompt = self._build_user_prompt(limited_stories, topics)
        
        # Make API call with retry logic
        start_time = time.time()
        
        try:
            response = await self._make_api_call_with_retry(system_prompt, user_prompt)
            summary = self._extract_summary(response)
            
            duration = time.time() - start_time
            
            logger.info(
                "Summary generated successfully",
                duration_seconds=round(duration, 2),
                summary_length=len(summary)
            )
            
            return summary
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Summary generation failed",
                duration_seconds=round(duration, 2),
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    async def _make_api_call_with_retry(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> ChatCompletion:
        """Make API call with exponential backoff retry logic.
        
        Args:
            system_prompt: System prompt for the AI
            user_prompt: User prompt with story content
            
        Returns:
            OpenAI ChatCompletion response
            
        Raises:
            SummarizerError: If all retries fail
        """
        max_retries = self.settings.openai_max_retries
        base_delay = 1.0
        max_delay = 60.0
        
        for attempt in range(max_retries + 1):
            try:
                response = await self._make_api_call(system_prompt, user_prompt)
                
                if attempt > 0:
                    logger.info("API call succeeded after retry", attempt=attempt + 1)
                
                return response
                
            except RateLimitError as e:
                if attempt == max_retries:
                    raise SummarizerError(
                        f"Rate limit exceeded after {max_retries + 1} attempts"
                    ) from e
                
                # Use retry_after if provided, otherwise exponential backoff
                if e.retry_after:
                    delay = min(e.retry_after, max_delay)
                else:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                
                logger.warning(
                    "Rate limit hit, retrying",
                    attempt=attempt + 1,
                    delay_seconds=delay,
                    retry_after=e.retry_after
                )
                
                await asyncio.sleep(delay)
                
            except (APIError, openai.APIError) as e:
                if attempt == max_retries:
                    raise SummarizerError(
                        f"API error after {max_retries + 1} attempts: {e}"
                    ) from e
                
                # Exponential backoff for API errors
                delay = min(base_delay * (2 ** attempt), max_delay)
                
                logger.warning(
                    "API error, retrying",
                    attempt=attempt + 1,
                    delay_seconds=delay,
                    error=str(e)
                )
                
                await asyncio.sleep(delay)
        
        raise SummarizerError("Should not reach here")
    
    async def _make_api_call(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> Any:
        """Make a single API call to OpenAI.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            OpenAI response (ChatCompletion for legacy, Response for GPT-5)
            
        Raises:
            RateLimitError: For rate limiting
            APIError: For other API errors
            TokenLimitError: For token limit exceeded
        """
        try:
            # Check if using GPT-5 models (new API)
            if self.settings.openai_model.startswith("gpt-5"):
                # Use new responses API for GPT-5
                combined_prompt = f"{system_prompt}\n\nUser Request:\n{user_prompt}"
                
                response = await self.client.responses.create(
                    model=self.settings.openai_model,
                    input=combined_prompt,
                    reasoning={"effort": "minimal"},
                    text={"verbosity": "medium"}
                )
                
                return response
            else:
                # Use legacy chat completions API for GPT-4 and earlier
                params = {
                    "model": self.settings.openai_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_completion_tokens": self.settings.openai_max_tokens,
                    "timeout": 30.0
                }
                
                # Only add temperature for non-nano models
                if self.settings.openai_model != "gpt-5-nano":
                    params["temperature"] = self.settings.openai_temperature
                
                response = await self.client.chat.completions.create(**params)
                
                return response
            
        except openai.RateLimitError as e:
            # Extract retry_after from headers if available
            retry_after = None
            if hasattr(e, 'response') and e.response:
                retry_after = e.response.headers.get('retry-after')
                if retry_after:
                    retry_after = float(retry_after)
            
            raise RateLimitError(str(e), retry_after=retry_after) from e
            
        except openai.BadRequestError as e:
            # Check if it's a token limit error
            if "maximum context length" in str(e).lower():
                raise TokenLimitError(f"Token limit exceeded: {e}") from e
            raise APIError(f"Bad request: {e}") from e
            
        except (openai.APIError, openai.OpenAIError) as e:
            raise APIError(f"OpenAI API error: {e}") from e
            
        except Exception as e:
            raise APIError(f"Unexpected error: {e}") from e
    
    def _build_system_prompt(self, topics: List[str]) -> str:
        """Build the system prompt for creating summaries.
        
        Args:
            topics: List of topics to focus on
            
        Returns:
            Simplified system prompt focused on summary quality
        """
        topics_str = ", ".join(topics)
        
        return f"""You are an expert tech journalist specializing in {topics_str}. Your task is to create engaging, insightful summaries of Hacker News stories.

All stories provided to you have already been pre-filtered for relevance to {topics_str}, so focus on creating high-quality summaries that provide value to readers.

REQUIRED FORMAT FOR EACH STORY:
```
**Title:** [Story title]
**What:** [1-3 sentences description of what this story is about]
**Why:** [1-3 sentences on why this is interesting/important for the topic]
**Link:** https://news.ycombinator.com/item?id=[story_id]
```

EXAMPLE:
```
🤖 AI/ML

**Title:** New LLM routing method achieves 93% GPT-4 performance at 25% cost
**What:** Research paper demonstrates a bandit-based routing system that selects optimal LLMs for tasks.
**Why:** Could significantly reduce AI deployment costs while maintaining quality, important for scaling AI applications.
**Link:** https://news.ycombinator.com/item?id=41234567

**Title:** Google AI Overview generates false story about user
**What:** User discovers Google's AI system created an elaborate false narrative about them.
**Why:** Highlights ongoing issues with AI hallucination and accuracy in user-facing AI products.
**Link:** https://news.ycombinator.com/item?id=41234568

💻 Programming

**Title:** New JavaScript framework for AI applications
**What:** Framework designed specifically for building AI-powered web applications with simplified ML integration.
**Why:** Addresses growing need for developer tools that make AI integration more accessible.
**Link:** https://news.ycombinator.com/item?id=41234569
```

FORMATTING REQUIREMENTS:
- Use section headers (like 🤖 AI/ML) only ONCE per topic, then list ALL related stories under that single header
- Use the exact format shown above for each story
- Maximum 15 stories total
- Focus on creating insightful "What" and "Why" explanations that provide real value to readers
- Group related stories by topic with appropriate emoji headers"""
    
    def _build_user_prompt(self, stories: List[Story], topics: List[str] = None) -> str:
        """Build the user prompt with story data.
        
        Args:
            stories: List of Story objects to summarize (already pre-filtered)
            topics: List of topics (used for reminders only)
            
        Returns:
            Formatted user prompt with story details
        """
        if topics is None:
            topics = []
        prompt = f"Here are {len(stories)} top Hacker News stories to summarize:\n\n"
        
        for i, story in enumerate(stories, 1):
            prompt += f"{i}. **{story.title}**\n"
            prompt += f"   🆔 HN ID: {story.id}\n"
            
            if story.url:
                prompt += f"   🔗 External URL: {story.url}\n"
            
            prompt += f"   📊 {story.score} points • {story.comment_count} comments • by {story.by}\n"
            prompt += f"   ⏰ {story.age_hours:.1f} hours ago\n"
            
            # Add text content for Ask HN, Show HN posts
            if story.is_text_post and story.text:
                # Truncate long text content
                text_preview = story.text[:200] + "..." if len(story.text) > 200 else story.text
                prompt += f"   💬 {text_preview}\n"
            
            prompt += "\n"
        
        prompt += f"""Please create a structured summary using the exact format specified in the system prompt.

KEY REMINDERS:
- Use section headers (🤖 AI/ML, 💻 Programming, etc.) only ONCE per topic, then list all related stories under that header
- For the **Link:** field, always use https://news.ycombinator.com/item?id=[HN_ID] format using the HN ID provided above
- Focus on creating valuable insights in the "What" and "Why" sections"""
        
        return prompt
    
    def _extract_summary(self, response) -> str:
        """Extract summary text from OpenAI response.
        
        Args:
            response: OpenAI response (ChatCompletion or Response)
            
        Returns:
            Summary text content
            
        Raises:
            SummarizerError: If response is invalid
        """
        # Handle GPT-5 responses API format
        if hasattr(response, 'output_text'):
            content = response.output_text
            if not content:
                raise SummarizerError("Empty output_text in GPT-5 response")
            content = content.strip()
            # Clean up any exclusion messages that slipped through
            content = self._clean_exclusion_messages(content)
            return content
        
        # Handle legacy chat completions format
        if not hasattr(response, 'choices') or not response.choices:
            raise SummarizerError("No response choices in OpenAI response")
        
        choice = response.choices[0]
        if not choice.message or not choice.message.content:
            raise SummarizerError("No content in OpenAI response")
        
        content = choice.message.content.strip()
        if not content:
            raise SummarizerError("Empty content in OpenAI response")
        
        # Clean up any exclusion messages that slipped through
        content = self._clean_exclusion_messages(content)
        
        return content
    
    def _clean_exclusion_messages(self, content: str) -> str:
        """Remove content that doesn't follow the structured format.
        
        Args:
            content: Raw AI output content
            
        Returns:
            Cleaned content with only properly structured stories
        """
        lines = content.split('\n')
        cleaned_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines (keep them for formatting)
            if not line:
                cleaned_lines.append('')
                i += 1
                continue
            
            # Check if this is a section header (emoji)
            if any(emoji in line for emoji in ['🤖', '💻', '🚀', '🔒', '⚡', '🌐', '💾', '🧭', '📊', '🔬']):
                cleaned_lines.append(line)
                i += 1
                continue
            
            # Check if this starts a properly structured story
            if line.startswith('**Title:**'):
                # Look ahead for the complete story structure
                story_lines = [line]
                j = i + 1
                has_what = False
                has_why = False
                has_link = False
                
                # Collect the next few lines to check for complete structure
                while j < len(lines) and j < i + 10:  # Look ahead max 10 lines
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    if next_line.startswith('**Title:**'):
                        # Hit next story, stop looking
                        break
                    
                    story_lines.append(next_line)
                    
                    if next_line.startswith('**What:**'):
                        has_what = True
                    elif next_line.startswith('**Why:**'):
                        has_why = True
                    elif next_line.startswith('**Link:**'):
                        has_link = True
                    
                    j += 1
                
                # Only include if it has the complete structure
                if has_what and has_why and has_link:
                    cleaned_lines.extend(story_lines)
                    i = j
                else:
                    # Skip malformed story
                    logger.debug(f"Filtering out malformed story: {line[:50]}...")
                    i = j
            else:
                # Regular line (not a story), keep it
                cleaned_lines.append(line)
                i += 1
        
        # Clean up extra empty lines
        result = '\n'.join(cleaned_lines).strip()
        while '\n\n\n' in result:
            result = result.replace('\n\n\n', '\n\n')
        
        return result
    

    
    async def close(self):
        """Close the OpenAI client connection."""
        await self.client.close()
        logger.info("StorySummarizer closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()