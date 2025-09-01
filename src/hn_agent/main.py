"""Main digest generation logic for Hacker News AI Summary Agent."""

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import structlog

from .config import get_settings, HNAgentSettings
from .hn_client import HackerNewsClient
from .models import Story, Summary
from .summarizer import StorySummarizer
from .slack_client import SlackClient

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@dataclass
class DigestStats:
    """Statistics from digest generation process."""
    
    total_fetched: int
    after_basic_filter: int
    after_topic_filter: int
    after_llm_filter: int
    after_hotness_filter: int
    summarized_count: int
    execution_time: float
    errors: List[str]
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_fetched == 0:
            return 0.0
        return (self.summarized_count / self.total_fetched) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/output."""
        return {
            "total_fetched": self.total_fetched,
            "after_basic_filter": self.after_basic_filter,
            "after_topic_filter": self.after_topic_filter,
            "after_llm_filter": self.after_llm_filter,
            "after_hotness_filter": self.after_hotness_filter,
            "summarized_count": self.summarized_count,
            "execution_time": round(self.execution_time, 2),
            "success_rate": round(self.success_rate, 1),
            "errors": self.errors
        }


class HackerNewsDigest:
    """Main digest generation orchestrator that coordinates all components."""
    
    def __init__(self, settings: HNAgentSettings):
        """Initialize digest generator with all required components.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.stats = DigestStats(
            total_fetched=0,
            after_basic_filter=0,
            after_topic_filter=0,
            after_llm_filter=0,
            after_hotness_filter=0,
            summarized_count=0,
            execution_time=0.0,
            errors=[]
        )
        
        # Initialize components
        self.hn_client = HackerNewsClient(
            base_url=str(settings.hn_api_base_url),
            timeout=settings.hn_timeout,
            max_concurrent_requests=settings.hn_max_concurrent,
            request_delay=settings.hn_request_delay
        )
        
        self.summarizer = StorySummarizer(settings)
        self.slack_client = SlackClient(settings)
        
        # Get topic keywords for filtering
        self.topic_keywords = settings.get_topic_keywords()
        
        logger.info(
            "HackerNewsDigest initialized",
            topics=list(settings.get_effective_topics()[:5]),  # Show first 5 topics
            topic_count=len(self.topic_keywords),
            max_stories=settings.max_stories,
            summary_max_stories=settings.summary_max_stories
        )
    
    async def generate_digest(self) -> Tuple[Optional[Summary], DigestStats]:
        """Generate complete digest with all pipeline stages.
        
        Returns:
            Tuple of (Summary object or None if failed, DigestStats)
        """
        start_time = time.time()
        
        logger.info(
            "Starting digest generation",
            max_stories=self.settings.max_stories,
            topics_count=len(self.topic_keywords)
        )
        
        try:
            # Stage 1: Fetch stories from HN
            stories = await self._fetch_stories()
            if not stories:
                self._add_error("No stories fetched from Hacker News")
                return None, self._finalize_stats(start_time)
            
            # Stage 2: Apply basic filtering (age, score, validity)
            filtered_stories = self._apply_basic_filters(stories)
            if not filtered_stories:
                self._add_error("No stories remained after basic filtering")
                return None, self._finalize_stats(start_time)
            
            # Stage 3: Apply topic filtering
            topic_filtered = self._apply_topic_filter(filtered_stories)
            if not topic_filtered:
                self._add_error("No stories matched topic filters")
                return None, self._finalize_stats(start_time)
            
            # Stage 3.5: LLM-based relevance filtering
            llm_filtered = await self._apply_llm_relevance_filter(topic_filtered)
            if not llm_filtered:
                self._add_error("No stories passed LLM relevance check")
                return None, self._finalize_stats(start_time)
            
            # Stage 4: Apply hotness scoring and ranking
            ranked_stories = self._apply_hotness_ranking(llm_filtered)
            
            # Stage 5: Generate summary
            summary = await self._generate_summary(ranked_stories)
            if not summary:
                self._add_error("Failed to generate summary")
                return None, self._finalize_stats(start_time)
            
            # Stage 6: Send to Slack (if not dry run)
            if not self.settings.dry_run:
                slack_success = await self._send_to_slack(summary, ranked_stories)
                if not slack_success:
                    self._add_error("Failed to send to Slack")
                    # Don't fail the entire process if Slack fails
            else:
                logger.info("DRY RUN: Skipping Slack notification")
                # Write summary to file for easy viewing
                with open("summary_output.txt", "w") as f:
                    f.write(summary.content)
                print("\n" + "="*80)
                print("DRY RUN: Summary saved to summary_output.txt")
                print("="*80 + "\n")
            
            self.stats.summarized_count = len(ranked_stories)
            
            logger.info(
                "Digest generation completed successfully",
                execution_time=time.time() - start_time,
                stories_processed=len(ranked_stories)
            )
            
            return summary, self._finalize_stats(start_time)
            
        except Exception as e:
            self._add_error(f"Unexpected error: {str(e)}")
            logger.error(
                "Digest generation failed",
                error=str(e),
                error_type=type(e).__name__,
                execution_time=time.time() - start_time
            )
            return None, self._finalize_stats(start_time)
    
    async def _fetch_stories(self) -> List[Story]:
        """Fetch stories from Hacker News API.
        
        Returns:
            List of Story objects
        """
        logger.info("Fetching stories from Hacker News API")
        
        try:
            stories = await self.hn_client.get_recent_stories(
                limit=self.settings.max_stories,
                max_age_hours=self.settings.max_age_hours
            )
            
            self.stats.total_fetched = len(stories)
            
            logger.info(
                "Stories fetched successfully",
                count=len(stories),
                max_age_hours=self.settings.max_age_hours
            )
            
            return stories
            
        except Exception as e:
            logger.error("Failed to fetch stories", error=str(e))
            raise
    
    def _apply_basic_filters(self, stories: List[Story]) -> List[Story]:
        """Apply basic filters (validity, score, age).
        
        Args:
            stories: Input stories
            
        Returns:
            Filtered stories
        """
        logger.info("Applying basic filters", input_count=len(stories))
        
        filtered = []
        
        for story in stories:
            # Skip invalid stories
            if not story.is_valid:
                continue
            
            # Skip low-scoring stories
            if story.score < self.settings.filter_min_score:
                continue
            
            # Skip stories that are too old
            if story.age_hours > self.settings.max_age_hours:
                continue
            
            # Skip stories with empty titles
            if not story.title or len(story.title.strip()) < 10:
                continue
            
            filtered.append(story)
        
        self.stats.after_basic_filter = len(filtered)
        
        logger.info(
            "Basic filtering completed",
            input_count=len(stories),
            output_count=len(filtered),
            filter_rate=f"{(len(filtered) / len(stories)) * 100:.1f}%"
        )
        
        return filtered
    
    def _apply_topic_filter(self, stories: List[Story]) -> List[Story]:
        """Apply topic-based filtering.
        
        Args:
            stories: Input stories
            
        Returns:
            Topic-filtered stories
        """
        logger.info("Applying topic filters", input_count=len(stories))
        
        filtered_stories = []
        topic_matches = {}
        excluded_stories = []
        
        for story in stories:
            matches = self._get_topic_matches(story)
            if matches:
                filtered_stories.append(story)
                for topic in matches:
                    topic_matches[topic] = topic_matches.get(topic, 0) + 1
            else:
                excluded_stories.append(story.title)
        
        self.stats.after_topic_filter = len(filtered_stories)
        
        logger.info(
            "Topic filtering completed",
            input_count=len(stories),
            output_count=len(filtered_stories),
            excluded_count=len(excluded_stories),
            topic_distribution=dict(list(topic_matches.items())[:10]),  # Show top 10
            sample_excluded=excluded_stories[:5] if excluded_stories else []  # Show sample of excluded stories
        )
        
        return filtered_stories
    
    def _get_topic_matches(self, story: Story) -> List[str]:
        """Get list of topics that match a story with enhanced filtering.
        
        Args:
            story: Story to check
            
        Returns:
            List of matching topic keywords
        """
        # First check for immediate exclusions
        if self._should_exclude_story(story):
            return []
        
        matches = []
        title_lower = story.title.lower()
        
        # Also check URL domain for context
        url_text = ""
        if story.url:
            url_text = story.url.lower()
        
        # Check text content for Ask HN/Show HN posts
        text_content = ""
        if story.text:
            text_content = story.text.lower()[:500]  # First 500 chars
        
        search_text = f"{title_lower} {url_text} {text_content}"
        
        # Apply keyword matching with stricter rules
        for keyword in self.topic_keywords:
            if self._keyword_matches_story(keyword, search_text, story):
                matches.append(keyword)
        
        return matches
    
    def _should_exclude_story(self, story: Story) -> bool:
        """Check if a story should be immediately excluded.
        
        Args:
            story: Story to check
            
        Returns:
            True if story should be excluded
        """
        title_lower = story.title.lower()
        
        # Explicit exclusion patterns
        exclusion_patterns = [
            # Hiring/recruitment posts
            "who is hiring",
            "who wants to be hired",
            "freelancer? seeking freelancer",
            "hiring thread",
            "jobs",
            
            # Ask HN posts that are typically off-topic
            "ask hn: what are you working on",
            "ask hn: what's the best",
            "ask hn: how do you",
            "ask hn: where do you",
            "ask hn: why do you",
            
            # Politics and geopolitics
            "politics",
            "election",
            "government",
            "policy",
            
            # Entertainment/culture
            "movie",
            "film",
            "music",
            "book",
            
            # Personal/lifestyle
            "personal",
            "life",
            "diet",
            "exercise",
        ]
        
        # Conditional exclusions that depend on context
        if "regulation" in title_lower and not any(tech in title_lower for tech in ["ai", "ml", "tech", "crypto"]):
            exclusion_patterns.append("regulation")
        
        if "art" in title_lower and "artificial" not in title_lower:
            exclusion_patterns.append("art")
            
        if "health" in title_lower and not any(tech in title_lower for tech in ["ai", "ml", "data", "tech"]):
            exclusion_patterns.append("health")
        
        # Check for exclusion patterns
        for pattern in exclusion_patterns:
            if pattern in title_lower:
                return True
        
        # Check if it's a generic Ask HN that doesn't mention our topics
        if title_lower.startswith("ask hn:"):
            # Allow if it explicitly mentions our focus topics
            focus_keywords = ["ai", "artificial intelligence", "machine learning", "ml", "llm", "gpt", "neural"]
            if not any(keyword in title_lower for keyword in focus_keywords):
                return True
        
        return False
    
    def _keyword_matches_story(self, keyword: str, search_text: str, story: Story) -> bool:
        """Check if a keyword matches a story with enhanced logic.
        
        Args:
            keyword: Keyword to check
            search_text: Combined text to search in
            story: Story object for additional context
            
        Returns:
            True if keyword matches story appropriately
        """
        if keyword not in search_text:
            return False
        
        # For AI/ML keywords, ensure they're not just tangential mentions
        if keyword.lower() in ["ai", "artificial intelligence", "machine learning", "ml"]:
            title_lower = story.title.lower()
            
            # Strong positive signals
            strong_signals = [
                "ai",
                "artificial intelligence", 
                "machine learning",
                "neural network",
                "deep learning",
                "llm",
                "gpt",
                "openai",
                "anthropic",
                "model",
                "training",
                "inference",
                "transformer",
                "embedding"
            ]
            
            if any(signal in title_lower for signal in strong_signals):
                return True
            
            # Weak signals that need more context
            weak_signals = ["data", "algorithm", "prediction", "automation"]
            if any(signal in title_lower for signal in weak_signals):
                # Only accept if there's additional AI context
                ai_context = ["ai", "ml", "machine", "learning", "intelligence", "neural"]
                return any(context in search_text for context in ai_context)
        
        return True
    
    async def _apply_llm_relevance_filter(self, stories: List[Story]) -> List[Story]:
        """Use LLM to check story relevance to selected topics.
        
        Args:
            stories: Stories to check for relevance
            
        Returns:
            List of stories that are relevant to the topics
        """
        if not stories:
            return []
        
        logger.info("Applying LLM relevance filtering", input_count=len(stories))
        
        relevant_stories = []
        topics_str = ", ".join(self.settings.get_effective_topics())
        
        # Process stories in batches to be more efficient
        batch_size = 5
        for i in range(0, len(stories), batch_size):
            batch = stories[i:i + batch_size]
            batch_results = await self._check_batch_relevance(batch, topics_str)
            relevant_stories.extend(batch_results)
        
        # Update stats
        self.stats.after_llm_filter = len(relevant_stories)
        
        logger.info(
            "LLM relevance filtering completed",
            input_count=len(stories),
            output_count=len(relevant_stories),
            filtered_out=len(stories) - len(relevant_stories)
        )
        
        return relevant_stories
    
    async def _check_batch_relevance(self, stories: List[Story], topics_str: str) -> List[Story]:
        """Check a batch of stories for relevance using LLM.
        
        Args:
            stories: Batch of stories to check
            topics_str: Topics string for filtering
            
        Returns:
            List of relevant stories from the batch
        """
        # Build prompt for batch checking
        system_prompt = f"""You are a content filter that determines if Hacker News stories are relevant to specific topics.

TOPICS TO MATCH: {topics_str}

Your task: For each story, respond with ONLY "YES" or "NO" - one word per line.

A story is relevant if it:
- Directly discusses the specified topics
- Is about technical developments, research, or products in these areas
- Reports news that impacts these technology sectors

A story is NOT relevant if it:
- Is about hiring/jobs (even if they mention the topics)
- Is general business news not specifically about the technologies
- Is about politics, culture, or other unrelated topics
- Only mentions the topics tangentially

Respond with exactly one word per story: "YES" or "NO"."""

        user_prompt = "Stories to evaluate:\n\n"
        for i, story in enumerate(stories, 1):
            user_prompt += f"{i}. {story.title}\n"
            if story.url:
                user_prompt += f"   URL: {story.url}\n"
            if story.text and len(story.text) > 0:
                # Include first 200 chars of text for Ask HN/Show HN posts
                preview = story.text[:200] + "..." if len(story.text) > 200 else story.text
                user_prompt += f"   Content: {preview}\n"
            user_prompt += "\n"
        
        user_prompt += f"\nFor each story (1-{len(stories)}), respond with exactly one word: YES or NO"
        
        try:
            # Use the summarizer's LLM client for consistency
            response = await self.summarizer._make_api_call(system_prompt, user_prompt)
            result_text = self.summarizer._extract_summary(response)
            
            # Parse responses
            responses = [line.strip().upper() for line in result_text.split('\n') if line.strip()]
            
            # Filter stories based on responses
            relevant_stories = []
            for i, story in enumerate(stories):
                if i < len(responses) and responses[i] == "YES":
                    relevant_stories.append(story)
                elif i < len(responses):
                    logger.debug(f"LLM filtered out story: {story.title[:50]}...")
            
            return relevant_stories
            
        except Exception as e:
            logger.error(f"LLM relevance check failed, keeping all stories: {e}")
            # Fallback: return all stories if LLM check fails
            return stories
    
    def _apply_hotness_ranking(self, stories: List[Story]) -> List[Story]:
        """Apply hotness scoring and rank stories by hotness.
        
        Args:
            stories: Input stories
            
        Returns:
            Stories ranked by hotness score (highest first)
        """
        logger.info("Calculating hotness scores", story_count=len(stories))
        
        scored_stories = []
        
        for story in stories:
            hotness_score = self._calculate_hotness_score(story)
            scored_stories.append((story, hotness_score))
        
        # Sort by hotness score (highest first)
        scored_stories.sort(key=lambda x: x[1], reverse=True)
        
        # Take top stories for summarization
        top_stories = [story for story, score in scored_stories[:self.settings.summary_max_stories]]
        
        self.stats.after_hotness_filter = len(top_stories)
        
        # Log hotness distribution
        if scored_stories:
            scores = [score for _, score in scored_stories]
            logger.info(
                "Hotness ranking completed",
                story_count=len(stories),
                selected_count=len(top_stories),
                max_hotness=round(max(scores), 2),
                min_hotness=round(min(scores), 2),
                avg_hotness=round(sum(scores) / len(scores), 2)
            )
        
        return top_stories
    
    def _calculate_hotness_score(self, story: Story) -> float:
        """Calculate hotness score for a story using HN-inspired algorithm.
        
        The algorithm considers:
        - Story score (upvotes)
        - Age decay (newer stories are hotter)
        - Comment engagement
        - Topic relevance boost
        - Story type considerations
        
        Based on Hacker News ranking algorithm with modifications.
        
        Args:
            story: Story to score
            
        Returns:
            Hotness score (higher = hotter)
        """
        # Base score from upvotes (subtract 1 for submitter's vote)
        base_score = max(story.score - 1, 0)
        
        # Age decay factor using HN's formula: (p-1) / (t+2)^gravity
        # where gravity controls how quickly stories lose hotness
        age_hours = max(story.age_hours, 0.1)  # Avoid division by zero
        gravity = 1.8
        age_penalty = math.pow(age_hours + 2, gravity)
        
        # Comment engagement boost (more comments = more interesting)
        comment_factor = 1 + (story.comment_count * 0.1)
        
        # Topic relevance boost (stories matching more topics get boost)
        topic_matches = self._get_topic_matches(story)
        topic_factor = 1 + (len(topic_matches) * 0.2)
        
        # Story type boost (Ask HN and Show HN often generate good discussions)
        type_factor = self._get_story_type_factor(story)
        
        # Recency boost for very fresh stories (first few hours are critical)
        recency_factor = 1.0
        if story.age_hours < 1.0:
            recency_factor = 2.0  # Big boost for very new stories
        elif story.age_hours < 3.0:
            recency_factor = 1.5
        elif story.age_hours < 6.0:
            recency_factor = 1.2
        
        # Calculate final hotness score
        hotness = (base_score * comment_factor * topic_factor * type_factor * recency_factor) / age_penalty
        
        return hotness
    
    def _get_story_type_factor(self, story: Story) -> float:
        """Get multiplier based on story type.
        
        Args:
            story: Story object
            
        Returns:
            Type-based multiplier
        """
        type_factors = {
            "ask": 1.3,    # Ask HN posts often generate good discussions
            "show": 1.2,   # Show HN posts are often interesting
            "story": 1.0,  # Regular stories (baseline)
        }
        return type_factors.get(story.type.value, 1.0)
    
    async def _generate_summary(self, stories: List[Story]) -> Optional[Summary]:
        """Generate AI summary of stories.
        
        Args:
            stories: Stories to summarize
            
        Returns:
            Summary object or None
        """
        if not stories:
            logger.warning("No stories to summarize")
            return None
        
        logger.info("Generating AI summary", story_count=len(stories))
        
        try:
            topics = list(self.settings.get_effective_topics())
            summary_text = await self.summarizer.summarize_stories(
                stories, topics
            )
            
            summary = Summary(
                content=summary_text,
                story_count=len(stories),
                topics=topics,
                generated_at=datetime.now(),
                total_score=sum(story.score for story in stories)
            )
            
            logger.info(
                "Summary generated successfully",
                summary_length=len(summary_text)
            )
            
            return summary
            
        except Exception as e:
            logger.error("Failed to generate summary", error=str(e))
            return None
    
    async def _send_to_slack(
        self, 
        summary: Summary, 
        stories: List[Story]
    ) -> bool:
        """Send summary to Slack.
        
        Args:
            summary: Summary to send
            stories: Stories that were summarized
            
        Returns:
            True if successful
        """
        logger.info("Sending summary to Slack")
        
        try:
            success = await self.slack_client.send_summary(summary, stories)
            
            if success:
                logger.info("Summary sent to Slack successfully")
            else:
                logger.error("Failed to send summary to Slack")
            
            return success
            
        except Exception as e:
            logger.error("Error sending to Slack", error=str(e))
            
            # Try to send error notification
            try:
                await self.slack_client.send_error_notification(
                    f"Failed to send digest: {str(e)}",
                    {"story_count": len(stories), "summary_length": len(summary.content)},
                    "error"
                )
            except:
                pass  # Don't fail if error notification fails too
            
            return False
    
    def _add_error(self, error: str) -> None:
        """Add error to stats tracking.
        
        Args:
            error: Error message
        """
        self.stats.errors.append(error)
        logger.error("Digest error", error=error)
    
    def _finalize_stats(self, start_time: float) -> DigestStats:
        """Finalize stats with execution time.
        
        Args:
            start_time: Process start time
            
        Returns:
            Completed stats object
        """
        self.stats.execution_time = time.time() - start_time
        
        logger.info(
            "Digest generation stats",
            **self.stats.to_dict()
        )
        
        return self.stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components.
        
        Returns:
            Health check results
        """
        logger.info("Performing health check")
        
        health = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "overall": "unknown"
        }
        
        # Check HN API
        try:
            test_stories = await self.hn_client.get_top_stories(limit=1)
            health["components"]["hn_api"] = {
                "status": "healthy" if test_stories else "degraded",
                "details": f"Fetched {len(test_stories)} stories"
            }
        except Exception as e:
            health["components"]["hn_api"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check OpenAI (simple validation)
        health["components"]["openai"] = {
            "status": "configured",
            "model": self.settings.openai_model
        }
        
        # Check Slack webhook (URL validation)
        health["components"]["slack"] = {
            "status": "configured",
            "webhook_domain": str(self.settings.slack_webhook_url).split('/')[2]
        }
        
        # Overall health
        component_statuses = [comp["status"] for comp in health["components"].values()]
        if all(status in ["healthy", "configured"] for status in component_statuses):
            health["overall"] = "healthy"
        elif any(status == "unhealthy" for status in component_statuses):
            health["overall"] = "unhealthy"
        else:
            health["overall"] = "degraded"
        
        logger.info("Health check completed", overall_status=health["overall"])
        
        return health
    
    async def close(self):
        """Close all component connections."""
        logger.info("Closing digest generator")
        
        try:
            await self.hn_client.close()
            await self.summarizer.close()
            await self.slack_client.close()
            logger.info("All components closed successfully")
        except Exception as e:
            logger.error("Error closing components", error=str(e))
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


async def main_digest() -> int:
    """Main digest generation function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        settings = get_settings()
        
        # Configure logging based on settings
        if settings.json_logs:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.JSONRenderer()
                ],
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
        
        async with HackerNewsDigest(settings) as digest:
            summary, stats = await digest.generate_digest()
            
            if summary:
                logger.info("Digest generation successful", stats=stats.to_dict())
                return 0
            else:
                logger.error("Digest generation failed", stats=stats.to_dict())
                return 1
                
    except Exception as e:
        logger.error("Fatal error in main", error=str(e), error_type=type(e).__name__)
        return 1


def create_cli() -> argparse.ArgumentParser:
    """Create command-line interface."""
    parser = argparse.ArgumentParser(
        description="Hacker News AI Summary Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate digest with default settings
  python -m hn_agent.main

  # Generate digest in dry-run mode (no Slack notification)
  python -m hn_agent.main --dry-run

  # Health check
  python -m hn_agent.main --health-check

  # Generate digest with custom topics
  TOPICS="AI,machine learning,startups" python -m hn_agent.main

  # Generate digest with debug logging
  LOG_LEVEL=DEBUG python -m hn_agent.main
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending Slack notifications"
    )
    
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform health check on all components"
    )
    
    parser.add_argument(
        "--topics",
        type=str,
        help="Comma-separated list of topics to filter for"
    )
    
    parser.add_argument(
        "--max-stories",
        type=int,
        help="Maximum number of stories to fetch"
    )
    
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Output structured JSON logs"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser


async def run_health_check() -> int:
    """Run health check and return exit code."""
    try:
        settings = get_settings()
        
        async with HackerNewsDigest(settings) as digest:
            health = await digest.health_check()
            
            # Output health check results
            print(json.dumps(health, indent=2))
            
            if health["overall"] == "healthy":
                return 0
            else:
                return 1
                
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return 1


async def main() -> int:
    """Main CLI entry point."""
    parser = create_cli()
    args = parser.parse_args()
    
    # Override settings with CLI arguments
    import os
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    if args.topics:
        os.environ["TOPICS"] = args.topics
    if args.max_stories:
        os.environ["MAX_STORIES"] = str(args.max_stories)
    if args.json_logs:
        os.environ["JSON_LOGS"] = "true"
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
    
    # Configure logging level
    import logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Route to appropriate function
    if args.health_check:
        return await run_health_check()
    else:
        return await main_digest()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        sys.exit(1)