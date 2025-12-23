# CRITICAL: Environment variables are set in docling_parser.py BEFORE any library imports
# For Lambda, also configure these in: Lambda Console -> Configuration -> Environment variables
# This ensures they're available before the container even starts

import json
import requests
import traceback
from docling_parser import DoclingParser, logger


def lambda_handler(event: dict, context):
    logger.info(f"Received event: {json.dumps(event)}")
    try:
        # Parse event - support both API Gateway (body) and direct invocation
        event_data = event
        if 'body' in event:
            # API Gateway format - parse body JSON string
            try:
                event_data = json.loads(event['body'])
            except (json.JSONDecodeError, TypeError):
                event_data = event
        
        # Validate presigned URL
        presigned_url = event_data.get('presignedUrl', '')
        if not presigned_url:
            logger.error("Missing presigned URL in event")
            return {
                'statusCode': 400,
                'body': 'Missing presigned URL parameter'
            }

        # Log the URL we're about to request (redacted for security)
        logger.info(f"Fetching content from presigned URL (first 20 chars): {presigned_url[:20]}...")

        # Get file content
        response = requests.get(presigned_url)
        response.raise_for_status()
        file_bytes = response.content

        # Log success and content length
        logger.info(f"Successfully downloaded content, size: {len(file_bytes)} bytes")

        # Process parameters
        is_image_present = event_data.get('isImagePresent', False)
        is_md_response = event_data.get('isMdResponse', True)

        # Initialize parser with bytes content directly
        parser = DoclingParser(
            bytes_content=file_bytes,
            is_image_present=is_image_present,
            is_md_response=is_md_response
        )

        logger.info(f"Detected document type: {parser.doc_type}")

        result = parser.parse_documents()
        return {
            'statusCode': 200,
            'body': result
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {
            'statusCode': 502,
            'body': f'Error fetching document: {str(e)}'
        }
    except Exception as e:
        # Get full stack trace for debugging
        stack_trace = traceback.format_exc()
        logger.error(f"Error processing document: {str(e)}\nStack trace: {stack_trace}")
        return {
            'statusCode': 500,
            'body': f'Processing error: {str(e)}'
        }
