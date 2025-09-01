"""AWS Lambda handler for the Hacker News AI Summary Agent."""

import asyncio
import json
from typing import Any, Dict

from src.hn_agent import HackerNewsAgent, get_config
from src.hn_agent.logging_config import configure_logging, get_logger


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entry point.
    
    Args:
        event: Lambda event data (e.g., EventBridge schedule, API Gateway, etc.)
        context: Lambda context object
        
    Returns:
        Lambda response with status and results
        
    TODO:
    - Handle different event sources (EventBridge, API Gateway, SQS)
    - Add proper error handling and response formatting
    - Implement timeout handling for Lambda limits
    - Add CloudWatch metrics and logging
    """
    # Run the async main function
    return asyncio.run(async_lambda_handler(event, context))


async def async_lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Async Lambda handler implementation.
    
    Args:
        event: Lambda event data
        context: Lambda context object
        
    Returns:
        Response dictionary with statusCode and body
    """
    try:
        # Load configuration (from environment variables in Lambda)
        config = get_config()
        
        # Configure logging for Lambda (CloudWatch)
        configure_logging(log_level=config.log_level, json_logs=True)
        logger = get_logger(__name__)
        
        logger.info(
            "Lambda execution started", 
            request_id=context.aws_request_id,
            event_source=event.get("source", "unknown"),
            topics=config.topics
        )
        
        # TODO: Initialize the agent
        # agent = HackerNewsAgent(
        #     openai_api_key=config.openai_api_key,
        #     slack_webhook_url=config.slack_webhook_url,
        #     topics=config.topics,
        #     max_stories=config.max_stories
        # )
        
        # try:
        #     # Run the summary cycle
        #     summary = await agent.run_summary_cycle()
        #     
        #     if summary:
        #         logger.info("Summary generated successfully", summary_id=summary.generated_at)
        #         return {
        #             "statusCode": 200,
        #             "body": json.dumps({
        #                 "message": "Summary generated and sent successfully",
        #                 "story_count": summary.story_count,
        #                 "topics": summary.topics,
        #                 "generated_at": summary.generated_at.isoformat()
        #             })
        #         }
        #     else:
        #         logger.error("Failed to generate summary")
        #         return {
        #             "statusCode": 500,
        #             "body": json.dumps({"error": "Failed to generate summary"})
        #         }
        #         
        # finally:
        #     await agent.close()
        
        # Placeholder response
        logger.info("Lambda execution completed successfully")
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Hacker News AI Summary Agent executed successfully",
                "topics": config.topics,
                "status": "placeholder_implementation"
            })
        }
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(
            "Lambda execution failed", 
            error=str(e),
            error_type=type(e).__name__,
            request_id=getattr(context, 'aws_request_id', 'unknown')
        )
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Internal server error",
                "message": str(e)
            })
        }


# For local testing of Lambda handler
if __name__ == "__main__":
    # Mock Lambda event and context for testing
    test_event = {
        "source": "aws.events",
        "detail-type": "Scheduled Event",
        "detail": {}
    }
    
    class MockContext:
        aws_request_id = "test-request-id"
        function_name = "test-function"
        function_version = "$LATEST"
        
        def get_remaining_time_in_millis(self):
            return 30000
    
    # Test the handler
    response = lambda_handler(test_event, MockContext())
    print(f"Lambda response: {response}")