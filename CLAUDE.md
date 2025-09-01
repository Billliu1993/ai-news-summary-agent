# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered news summary agent designed to summarize selected topics from Hacker News. The agent fetches stories, filters by topics of interest (AI, programming, startups), generates summaries using OpenAI, and sends them to Slack.

## Development Environment

- **Python Version**: 3.12 (specified in .python-version)
- **Package Management**: Uses pyproject.toml with modern Python packaging
- **Package Structure**: Source code in `src/hn_agent/` following src-layout
- **Dependencies**: httpx, openai, pydantic, python-dotenv, structlog

## Common Commands

```bash
# Install dependencies
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Run the application (multiple ways)
python main.py                    # Old way (still works)
python -m hn_agent                # Python module execution
hn-agent                          # Console script (after pip install)
hn-digest                         # Alternative console script

# Run with options
hn-agent --dry-run                # Test without sending to Slack
hn-agent --health-check           # Check component health
hn-agent --debug --json-logs      # Debug mode with JSON output
hn-agent --topics "AI,startups"   # Custom topics

# Run linting and formatting
black src/ tests/
ruff check src/ tests/
mypy src/

# Run tests (when implemented)
pytest
```

## Configuration

Copy `.env.example` to `.env` and configure:
- `OPENAI_API_KEY`: OpenAI API key for summarization
- `SLACK_WEBHOOK_URL`: Slack webhook for notifications
- `TOPICS`: Comma-separated topics to filter (default: AI,programming,startups)

## Architecture

### Package Structure (`src/hn_agent/`)

- **`main.py`**: Main digest generation orchestrator with complete pipeline
- **`hn_client.py`**: Hacker News API client for fetching stories
- **`summarizer.py`**: OpenAI-powered story summarization with cost tracking
- **`slack_client.py`**: Rich Slack webhook integration with blocks formatting
- **`models.py`**: Pydantic data models (Story, Summary, etc.)
- **`config.py`**: Comprehensive Pydantic settings management
- **`__init__.py`**: Package exports and convenience functions
- **`__main__.py`**: Module execution support

### Pipeline Flow

1. Fetch top stories from Hacker News API
2. Filter stories based on configured topics
3. Generate AI summary using OpenAI
4. Format and send summary to Slack
5. Handle errors and logging throughout

## Deployment Options

### Local Development
```bash
# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Install and run
pip install -e .
python main.py
```

### AWS Lambda Deployment
- **Handler**: `lambda_handler.py` contains the Lambda entry point
- **Environment**: Configure via Lambda environment variables (same as .env)
- **Runtime**: Python 3.11+ (Lambda compatible)
- **Packaging**: Use your preferred method (Terraform, AWS CLI, etc.)

The Lambda handler includes:
- Proper async/await handling
- CloudWatch logging integration
- Error handling and response formatting
- Local testing capability

## Implementation Notes

Most components are currently placeholder implementations with TODO comments indicating what needs to be implemented. Each module has:
- Proper type hints and modern Python practices
- Async/await support where beneficial
- Structured logging with context
- Error handling and retry logic patterns
- Configuration through environment variables
- Lambda deployment compatibility