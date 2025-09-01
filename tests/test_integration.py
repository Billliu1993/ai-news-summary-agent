"""Integration tests for the full HN Agent pipeline."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.hn_agent.main import DigestStats, HackerNewsDigest
from src.hn_agent.models import Story, Summary
from tests.conftest import create_story_with_overrides


class TestHackerNewsDigestIntegration:
    """Integration tests for the complete digest pipeline."""

    @pytest.fixture
    def integration_settings(self, mock_settings):
        """Settings configured for integration testing."""
        mock_settings.max_stories = 20
        mock_settings.summary_max_stories = 5
        mock_settings.filter_min_score = 10
        mock_settings.max_age_hours = 24.0
        mock_settings.topics = "AI,programming,startups"
        mock_settings.dry_run = True
        return mock_settings

    @pytest.fixture
    def mock_stories_data(self):
        """Complete set of mock stories for integration testing."""
        return [
            # AI-related stories
            {
                "by": "aiuser1",
                "descendants": 45,
                "id": 1001,
                "kids": [1011, 1012],
                "score": 156,
                "time": 1704067200,
                "title": "New breakthrough in large language models",
                "type": "story",
                "url": "https://ai-research.com/breakthrough",
                "dead": False,
                "deleted": False,
            },
            {
                "by": "aiuser2",
                "descendants": 23,
                "id": 1002,
                "kids": [1021],
                "score": 89,
                "time": 1704070800,
                "title": "Ask HN: What AI tools do you actually use daily?",
                "type": "story",
                "url": None,
                "text": "I'm curious about practical AI applications.",
                "dead": False,
                "deleted": False,
            },
            # Programming stories
            {
                "by": "devuser1",
                "descendants": 67,
                "id": 1003,
                "kids": [1031, 1032, 1033],
                "score": 234,
                "time": 1704064800,
                "title": "Show HN: New programming language for concurrent systems",
                "type": "story",
                "url": "https://github.com/example/newlang",
                "dead": False,
                "deleted": False,
            },
            {
                "by": "devuser2",
                "descendants": 12,
                "id": 1004,
                "kids": [1041],
                "score": 78,
                "time": 1704071400,
                "title": "The evolution of programming paradigms in 2024",
                "type": "story",
                "url": "https://programming-blog.com/evolution",
                "dead": False,
                "deleted": False,
            },
            # Startup stories
            {
                "by": "startupuser",
                "descendants": 34,
                "id": 1005,
                "kids": [1051, 1052],
                "score": 145,
                "time": 1704068400,
                "title": "Startup raises $50M Series B for developer tools",
                "type": "story",
                "url": "https://techcrunch.com/startup-funding",
                "dead": False,
                "deleted": False,
            },
            # Low-scored story (should be filtered out)
            {
                "by": "lowuser",
                "descendants": 2,
                "id": 1006,
                "kids": [],
                "score": 5,
                "time": 1704069000,
                "title": "Small update to obscure library",
                "type": "story",
                "url": "https://github.com/obscure/update",
                "dead": False,
                "deleted": False,
            },
            # Off-topic story (should be filtered out)
            {
                "by": "offuser",
                "descendants": 15,
                "id": 1007,
                "kids": [1071],
                "score": 67,
                "time": 1704069600,
                "title": "Best hiking trails in California",
                "type": "story",
                "url": "https://hiking-guide.com/california",
                "dead": False,
                "deleted": False,
            },
            # Old story (should be filtered out)
            {
                "by": "olduser",
                "descendants": 56,
                "id": 1008,
                "kids": [1081, 1082],
                "score": 189,
                "time": 1704000000,
                "title": "AI breakthrough from last week",  # Too old
                "type": "story",
                "url": "https://old-news.com/ai",
                "dead": False,
                "deleted": False,
            },
        ]

    @pytest.fixture
    def expected_summary_content(self):
        """Expected summary content from the integration test."""
        return """🤖 AI/ML

**Title:** New breakthrough in large language models
**What:** Research demonstrates significant advances in language model capabilities and efficiency.
**Why:** This could enable more powerful AI applications across multiple industries.
**Link:** https://news.ycombinator.com/item?id=1001

**Title:** Ask HN: What AI tools do you actually use daily?
**What:** Community discussion about practical AI applications in everyday work.
**Why:** Shows the growing integration of AI tools in professional workflows.
**Link:** https://news.ycombinator.com/item?id=1002

💻 Programming

**Title:** Show HN: New programming language for concurrent systems
**What:** Developer introduces a new language designed for high-performance concurrent programming.
**Why:** Addresses growing need for better concurrency tools in modern software development.
**Link:** https://news.ycombinator.com/item?id=1003

**Title:** The evolution of programming paradigms in 2024
**What:** Analysis of how programming approaches have changed and evolved this year.
**Why:** Helps developers understand trends shaping the future of software development.
**Link:** https://news.ycombinator.com/item?id=1004

🚀 Startups

**Title:** Startup raises $50M Series B for developer tools
**What:** Company secures significant funding to expand developer productivity platform.
**Why:** Indicates strong investor confidence in the developer tooling market.
**Link:** https://news.ycombinator.com/item?id=1005"""

    def setup_mocks(self, digest, mock_stories_data, expected_summary_content):
        """Set up comprehensive mocks for integration testing."""

        # Mock HN Client
        async def mock_get_top_stories(limit):
            return [story["id"] for story in mock_stories_data[:limit]]

        async def mock_get_recent_stories(limit, max_age_hours):
            # Filter by age (stories after 1704060000 are recent)
            recent_cutoff = 1704060000
            recent_stories = [
                Story(**story)
                for story in mock_stories_data
                if story["time"] > recent_cutoff
            ]
            return recent_stories[:limit]

        digest.hn_client.get_top_stories = AsyncMock(side_effect=mock_get_top_stories)
        digest.hn_client.get_recent_stories = AsyncMock(
            side_effect=mock_get_recent_stories
        )

        # Mock Summarizer
        async def mock_summarize_stories(stories, topics):
            # Return expected summary content
            return expected_summary_content

        # Mock LLM relevance filter
        async def mock_make_api_call(system_prompt, user_prompt):
            # Simulate LLM filtering - return YES for on-topic stories
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]

            # Count stories in the prompt and return appropriate YES/NO responses
            story_count = user_prompt.count(". ")  # Rough count of stories
            responses = []
            for i in range(story_count):
                # Reject off-topic stories (hiking) and low-quality stories
                if i == 5:  # Off-topic story index
                    responses.append("NO")
                else:
                    responses.append("YES")

            mock_response.choices[0].message.content = "\n".join(responses)
            return mock_response

        digest.summarizer.summarize_stories = AsyncMock(
            side_effect=mock_summarize_stories
        )
        digest.summarizer._make_api_call = AsyncMock(side_effect=mock_make_api_call)
        digest.summarizer._extract_summary = lambda x: x.choices[0].message.content

        # Mock Slack Client
        async def mock_send_summary(summary, stories):
            return True

        digest.slack_client.send_summary = AsyncMock(side_effect=mock_send_summary)

    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self, integration_settings, mock_stories_data, expected_summary_content
    ):
        """Test the complete digest generation pipeline."""

        async with HackerNewsDigest(integration_settings) as digest:
            self.setup_mocks(digest, mock_stories_data, expected_summary_content)

            summary, stats = await digest.generate_digest()

            # Verify successful completion
            assert summary is not None
            assert isinstance(summary, Summary)
            assert isinstance(stats, DigestStats)

            # Verify summary properties
            assert len(summary.topics) > 0
            assert summary.story_count > 0
            assert summary.content == expected_summary_content

            # Verify stats tracking
            assert stats.total_fetched > 0
            assert stats.after_basic_filter > 0
            assert stats.after_topic_filter > 0
            assert stats.summarized_count > 0
            assert stats.execution_time > 0
            assert len(stats.errors) == 0

            # Verify filtering worked correctly
            assert (
                stats.after_basic_filter < stats.total_fetched
            )  # Some filtered by score/age
            assert (
                stats.after_topic_filter <= stats.after_basic_filter
            )  # Some filtered by topic

    @pytest.mark.asyncio
    async def test_pipeline_with_no_stories_fetched(self, integration_settings):
        """Test pipeline behavior when no stories are fetched."""

        async with HackerNewsDigest(integration_settings) as digest:
            # Mock empty story response
            digest.hn_client.get_recent_stories = AsyncMock(return_value=[])

            summary, stats = await digest.generate_digest()

            # Should fail gracefully
            assert summary is None
            assert stats.total_fetched == 0
            assert len(stats.errors) > 0
            assert "No stories fetched" in str(stats.errors)

    @pytest.mark.asyncio
    async def test_pipeline_with_basic_filtering_removes_all(
        self, integration_settings, mock_stories_data
    ):
        """Test pipeline when basic filtering removes all stories."""

        async with HackerNewsDigest(integration_settings) as digest:
            # Create stories that will all be filtered out (low scores, old)
            bad_stories = [
                create_story_with_overrides(
                    id=i,
                    score=1,  # Below min score
                    time=1704000000,  # Too old
                    title=f"Bad story {i}",
                )
                for i in range(5)
            ]

            digest.hn_client.get_recent_stories = AsyncMock(return_value=bad_stories)

            summary, stats = await digest.generate_digest()

            assert summary is None
            assert stats.after_basic_filter == 0
            assert len(stats.errors) > 0
            assert "No stories remained after basic filtering" in str(stats.errors)

    @pytest.mark.asyncio
    async def test_pipeline_with_topic_filtering_removes_all(
        self, integration_settings
    ):
        """Test pipeline when topic filtering removes all stories."""

        async with HackerNewsDigest(integration_settings) as digest:
            # Create stories that don't match any topics
            off_topic_stories = [
                create_story_with_overrides(
                    id=i,
                    title=f"Unrelated story about cooking recipe {i}",
                    score=50,  # Good score but off-topic
                )
                for i in range(5)
            ]

            digest.hn_client.get_recent_stories = AsyncMock(
                return_value=off_topic_stories
            )

            summary, stats = await digest.generate_digest()

            assert summary is None
            assert stats.after_topic_filter == 0
            assert len(stats.errors) > 0
            assert "No stories matched topic filters" in str(stats.errors)

    @pytest.mark.asyncio
    async def test_pipeline_with_llm_filtering_removes_all(
        self, integration_settings, mock_stories_data
    ):
        """Test pipeline when LLM filtering removes all stories."""

        async with HackerNewsDigest(integration_settings) as digest:
            # Set up mocks but make LLM filter reject everything
            self.setup_mocks(digest, mock_stories_data, "")

            # Override LLM filter to reject all stories
            async def mock_make_api_call_reject_all(system_prompt, user_prompt):
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                story_count = user_prompt.count(". ")
                mock_response.choices[0].message.content = "\n".join(
                    ["NO"] * story_count
                )
                return mock_response

            digest.summarizer._make_api_call = AsyncMock(
                side_effect=mock_make_api_call_reject_all
            )

            summary, stats = await digest.generate_digest()

            assert summary is None
            assert stats.after_llm_filter == 0
            assert len(stats.errors) > 0
            assert "No stories passed LLM relevance check" in str(stats.errors)

    @pytest.mark.asyncio
    async def test_pipeline_with_summarization_failure(
        self, integration_settings, mock_stories_data
    ):
        """Test pipeline when summarization fails."""

        async with HackerNewsDigest(integration_settings) as digest:
            self.setup_mocks(digest, mock_stories_data, "")

            # Make summarization fail
            digest.summarizer.summarize_stories = AsyncMock(
                side_effect=Exception("Summarization failed")
            )

            summary, stats = await digest.generate_digest()

            assert summary is None
            assert len(stats.errors) > 0
            assert "Failed to generate summary" in str(stats.errors)

    @pytest.mark.asyncio
    async def test_pipeline_with_slack_failure(
        self, integration_settings, mock_stories_data, expected_summary_content
    ):
        """Test pipeline when Slack sending fails."""

        integration_settings.dry_run = False  # Enable Slack sending

        async with HackerNewsDigest(integration_settings) as digest:
            self.setup_mocks(digest, mock_stories_data, expected_summary_content)

            # Make Slack sending fail
            digest.slack_client.send_summary = AsyncMock(return_value=False)

            summary, stats = await digest.generate_digest()

            # Should still succeed but with error recorded
            assert summary is not None
            assert len(stats.errors) > 0
            assert "Failed to send to Slack" in str(stats.errors)

    @pytest.mark.asyncio
    async def test_pipeline_with_unexpected_error(
        self, integration_settings, mock_stories_data
    ):
        """Test pipeline with unexpected error during processing."""

        async with HackerNewsDigest(integration_settings) as digest:
            # Make story fetching raise unexpected error
            digest.hn_client.get_recent_stories = AsyncMock(
                side_effect=RuntimeError("Unexpected error")
            )

            summary, stats = await digest.generate_digest()

            assert summary is None
            assert len(stats.errors) > 0
            assert "Unexpected error" in str(stats.errors)
            assert stats.execution_time > 0  # Should still track timing

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, integration_settings):
        """Test health check when all components are healthy."""

        async with HackerNewsDigest(integration_settings) as digest:
            # Mock healthy responses
            digest.hn_client.get_top_stories = AsyncMock(return_value=[1, 2, 3])

            health = await digest.health_check()

            assert health["overall"] == "healthy"
            assert "hn_api" in health["components"]
            assert "openai" in health["components"]
            assert "slack" in health["components"]

            # HN API should be healthy
            assert health["components"]["hn_api"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_with_hn_api_failure(self, integration_settings):
        """Test health check when HN API fails."""

        async with HackerNewsDigest(integration_settings) as digest:
            # Mock HN API failure
            digest.hn_client.get_top_stories = AsyncMock(
                side_effect=Exception("API Error")
            )

            health = await digest.health_check()

            assert health["overall"] == "unhealthy"
            assert health["components"]["hn_api"]["status"] == "unhealthy"
            assert "API Error" in health["components"]["hn_api"]["error"]

    @pytest.mark.asyncio
    async def test_pipeline_stats_tracking(
        self, integration_settings, mock_stories_data, expected_summary_content
    ):
        """Test that pipeline properly tracks statistics at each stage."""

        async with HackerNewsDigest(integration_settings) as digest:
            self.setup_mocks(digest, mock_stories_data, expected_summary_content)

            summary, stats = await digest.generate_digest()

            assert summary is not None

            # Verify all stats are populated
            assert stats.total_fetched > 0
            assert stats.after_basic_filter > 0
            assert stats.after_topic_filter > 0
            assert stats.after_llm_filter > 0
            assert stats.after_hotness_filter > 0
            assert stats.summarized_count > 0
            assert stats.execution_time > 0

            # Verify filtering progression makes sense
            assert stats.after_basic_filter <= stats.total_fetched
            assert stats.after_topic_filter <= stats.after_basic_filter
            assert stats.after_llm_filter <= stats.after_topic_filter
            assert stats.after_hotness_filter <= stats.after_llm_filter
            assert stats.summarized_count <= stats.after_hotness_filter

            # Verify success rate calculation
            expected_rate = (stats.summarized_count / stats.total_fetched) * 100
            assert abs(stats.success_rate - expected_rate) < 0.01

    @pytest.mark.asyncio
    async def test_pipeline_hotness_ranking(
        self, integration_settings, mock_stories_data, expected_summary_content
    ):
        """Test that pipeline properly ranks stories by hotness."""

        async with HackerNewsDigest(integration_settings) as digest:
            self.setup_mocks(digest, mock_stories_data, expected_summary_content)

            # Override to capture the stories passed to summarizer
            captured_stories = []

            async def mock_summarize_and_capture(stories, topics):
                captured_stories.extend(stories)
                return expected_summary_content

            digest.summarizer.summarize_stories = AsyncMock(
                side_effect=mock_summarize_and_capture
            )

            summary, stats = await digest.generate_digest()

            assert summary is not None
            assert len(captured_stories) > 0

            # Verify stories are ordered by hotness (higher scores should generally come first)
            # Note: Exact ordering depends on hotness algorithm, but we can check basic properties
            scores = [story.score for story in captured_stories]
            assert len(scores) > 1

            # Should not include very low-scoring stories
            assert all(
                score >= integration_settings.filter_min_score for score in scores
            )

    @pytest.mark.asyncio
    async def test_pipeline_dry_run_mode(
        self, integration_settings, mock_stories_data, expected_summary_content
    ):
        """Test pipeline in dry run mode (no Slack sending)."""

        integration_settings.dry_run = True

        async with HackerNewsDigest(integration_settings) as digest:
            self.setup_mocks(digest, mock_stories_data, expected_summary_content)

            # Slack client should not be called in dry run
            slack_send_spy = AsyncMock()
            digest.slack_client.send_summary = slack_send_spy

            summary, stats = await digest.generate_digest()

            assert summary is not None
            assert stats.execution_time > 0

            # Slack should not be called in dry run mode
            slack_send_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_concurrent_safety(
        self, integration_settings, mock_stories_data, expected_summary_content
    ):
        """Test that running multiple digest instances concurrently works safely."""

        async def run_digest():
            async with HackerNewsDigest(integration_settings) as digest:
                self.setup_mocks(digest, mock_stories_data, expected_summary_content)
                summary, stats = await digest.generate_digest()
                return summary, stats

        # Run multiple digests concurrently
        tasks = [run_digest() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for result in results:
            assert not isinstance(result, Exception), f"Got exception: {result}"
            summary, stats = result
            assert summary is not None
            assert stats.summarized_count > 0

    @pytest.mark.asyncio
    async def test_pipeline_memory_cleanup(
        self, integration_settings, mock_stories_data, expected_summary_content
    ):
        """Test that pipeline properly cleans up resources."""

        digest = HackerNewsDigest(integration_settings)
        self.setup_mocks(digest, mock_stories_data, expected_summary_content)

        # Run digest
        summary, stats = await digest.generate_digest()
        assert summary is not None

        # Manually close (normally done by context manager)
        await digest.close()

        # Verify cleanup was called on components
        # Note: This test verifies the close method exists and can be called
        # In practice, the actual HTTP clients would be closed

    def test_digest_stats_dict_conversion(self):
        """Test DigestStats conversion to dictionary."""
        stats = DigestStats(
            total_fetched=100,
            after_basic_filter=80,
            after_topic_filter=60,
            after_llm_filter=45,
            after_hotness_filter=30,
            summarized_count=25,
            execution_time=5.75,
            errors=["Test error"],
        )

        stats_dict = stats.to_dict()

        assert stats_dict["total_fetched"] == 100
        assert stats_dict["summarized_count"] == 25
        assert stats_dict["execution_time"] == 5.75
        assert stats_dict["success_rate"] == 25.0  # 25/100 * 100
        assert stats_dict["errors"] == ["Test error"]

    def test_digest_stats_success_rate_edge_cases(self):
        """Test DigestStats success rate calculation edge cases."""
        # Zero fetched stories
        stats_zero = DigestStats(
            total_fetched=0,
            after_basic_filter=0,
            after_topic_filter=0,
            after_llm_filter=0,
            after_hotness_filter=0,
            summarized_count=0,
            execution_time=0.0,
            errors=[],
        )
        assert stats_zero.success_rate == 0.0

        # Perfect success rate
        stats_perfect = DigestStats(
            total_fetched=10,
            after_basic_filter=10,
            after_topic_filter=10,
            after_llm_filter=10,
            after_hotness_filter=10,
            summarized_count=10,
            execution_time=1.0,
            errors=[],
        )
        assert stats_perfect.success_rate == 100.0
