# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered news summary agent designed to summarize selected topics from Hacker News. The agent fetches stories from HN, applies multi-stage filtering (basic, topic-based, LLM relevance), generates summaries using OpenAI, and sends formatted messages to Slack.

## Development Environment

- **Python Version**: 3.12 (specified in .python-version)
- **Package Management**: Uses pyproject.toml with modern Python packaging
- **Package Structure**: Source code in `src/hn_agent/` following src-layout
- **Dependencies**: httpx, openai, pydantic, pydantic-settings, python-dotenv, structlog
- **Testing**: pytest with comprehensive test coverage

## Common Commands

```bash
# Install dependencies
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Run the application (multiple ways)
python main.py                    # Root entry point
python -m hn_agent.main           # Module execution
hn-agent                          # Console script (after pip install)
hn-digest                         # Alternative console script

# Run with options
hn-agent --dry-run                # Test without sending to Slack
hn-agent --health-check           # Check component health
hn-agent --debug --json-logs      # Debug mode with JSON output
hn-agent --topics "AI,startups"   # Custom topics
hn-agent --max-stories 100        # Override story limit

# Run linting and formatting
black src/ tests/
ruff check src/ tests/
mypy src/

# Run tests
pytest
pytest -v --cov=src/hn_agent      # With coverage
```

## Configuration

Copy `.env.example` to `.env` and configure required settings:

### Required Settings
- `OPENAI_API_KEY`: OpenAI API key (must start with 'sk-')
- `SLACK_WEBHOOK_URL`: Slack webhook URL

### Key Optional Settings
- `OPENAI_MODEL`: Model to use (default: gpt-5-nano)
- `TOPICS`: Comma-separated topics (default: AI,programming,startups)
- `MAX_STORIES`: Max stories to fetch (default: 50)
- `SUMMARY_MAX_STORIES`: Max stories in summary (default: 15)
- `DRY_RUN`: Test mode without Slack notification
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Architecture

### Package Structure (`src/hn_agent/`)

- **`main.py`**: Complete digest generation pipeline with multi-stage filtering
- **`hn_client.py`**: Async HN API client with rate limiting and error handling
- **`summarizer.py`**: OpenAI integration with retry logic and cost tracking
- **`slack_client.py`**: Rich Slack webhook client with block formatting
- **`models.py`**: Pydantic data models (Story, Summary, Comment, SlackMessage)
- **`config.py`**: Comprehensive Pydantic settings with validation
- **`__init__.py`**: Package exports and convenience functions
- **`__main__.py`**: Module execution support

### Complete Pipeline Flow

1. **Fetch Stories**: Get recent stories from HN API with rate limiting
2. **Basic Filtering**: Remove invalid, old, low-scoring stories
3. **Topic Filtering**: Match stories against configured topics with keyword matching
4. **LLM Relevance**: Use OpenAI to verify story relevance to topics
5. **Hotness Ranking**: Score stories using HN-inspired algorithm (score, age, comments)
6. **AI Summarization**: Generate comprehensive summary using OpenAI
7. **Slack Delivery**: Send formatted message with rich blocks to Slack

### Key Features

- **Multi-stage Filtering**: Progressive filtering with detailed stats tracking
- **LLM Relevance Checking**: Uses OpenAI to validate topic relevance
- **Hotness Algorithm**: HN-inspired scoring for story ranking
- **Rich Slack Formatting**: Block-based messages with story links and metadata
- **Comprehensive Error Handling**: Graceful failures with error notifications
- **Health Checks**: Component validation and status reporting
- **Dry Run Mode**: Test execution without external notifications

## Deployment Options

### Local Development
```bash
# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Install and run
pip install -e .
hn-agent --dry-run    # Test run
hn-agent              # Full run
```

### AWS Lambda Deployment
- **Handler**: `lambda_handler.py` contains Lambda entry point (currently placeholder)
- **Environment**: Configure via Lambda environment variables
- **Runtime**: Python 3.12+ compatible
- **Packaging**: Standard Lambda deployment methods

### Production Considerations
- Use `ENVIRONMENT=production` to enable production validations
- Set `JSON_LOGS=true` for structured logging
- Configure appropriate `LOG_LEVEL` and monitoring
- Set reasonable rate limits for HN API (`HN_REQUEST_DELAY`, `HN_MAX_CONCURRENT`)

## Testing

Comprehensive test suite with:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Mock Services**: HN API and OpenAI mocking for reliable tests
- **Configuration**: pytest.ini and conftest.py for test setup

Run tests:
```bash
pytest                           # All tests
pytest tests/test_hn_client.py   # Specific component
pytest -k "test_filtering"       # Specific functionality
```