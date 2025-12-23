# CRITICAL: Environment variables are set in docling_parser.py BEFORE any library imports
# For Lambda, also configure these in: Lambda Console -> Configuration -> Environment variables
# This ensures they're available before the container even starts

import json
import os
import uuid
from datetime import datetime
from urllib.parse import urlparse

import boto3
import requests
import traceback
from docling_parser import DoclingParser, logger

# Initialize S3 client
s3_client = boto3.client('s3')


def extract_filename_from_url(url: str) -> str:
    """Extract filename from URL path"""
    parsed = urlparse(url)
    path = parsed.path
    filename = path.split('/')[-1] if path else 'document'
    # Remove extension if present
    if '.' in filename:
        filename = filename.rsplit('.', 1)[0]
    return filename


def upload_to_s3(content: str, bucket: str, key: str) -> str:
    """Upload content to S3 and return the S3 URI"""
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=content.encode('utf-8'),
        ContentType='text/markdown'
    )
    return f"s3://{bucket}/{key}"


def generate_presigned_url(bucket: str, key: str, expiration: int = 3600) -> str:
    """Generate a presigned URL for viewing the file"""
    return s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=expiration
    )


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
                'body': json.dumps({'error': 'Missing presigned URL parameter'})
            }

        # Log the URL we're about to request (redacted for security)
        logger.info(f"Fetching content from presigned URL (first 50 chars): {presigned_url[:50]}...")

        # Get file content
        response = requests.get(presigned_url)
        response.raise_for_status()
        file_bytes = response.content

        # Log success and content length
        logger.info(f"Successfully downloaded content, size: {len(file_bytes)} bytes")

        # Processing parameters
        # Can be set via event params or Lambda environment variables
        is_md_response = event_data.get('isMdResponse', True)
        enable_table_extraction = event_data.get('enableTableExtraction', 
            os.environ.get('ENABLE_TABLE_EXTRACTION', 'true').lower() == 'true')
        enable_ocr = event_data.get('enableOcr', 
            os.environ.get('ENABLE_OCR', 'false').lower() == 'true')
        # Legacy support
        is_image_present = event_data.get('isImagePresent', False)
        
        # S3 output configuration (optional)
        output_bucket = event_data.get('outputBucket') or os.environ.get('OUTPUT_BUCKET')
        output_prefix = event_data.get('outputPrefix') or os.environ.get('OUTPUT_PREFIX', 'docling-output')
        generate_url = event_data.get('generatePresignedUrl', True)
        url_expiration = event_data.get('urlExpiration', 3600)  # 1 hour default

        # Initialize parser with bytes content directly
        parser = DoclingParser(
            bytes_content=file_bytes,
            is_image_present=is_image_present,
            is_md_response=is_md_response,
            enable_table_extraction=enable_table_extraction,
            enable_ocr=enable_ocr
        )

        logger.info(f"Detected document type: {parser.doc_type}")

        # Parse entire document (not page by page)
        result = parser.parse_documents()
        
        # Build response
        response_data = {
            'success': True,
            'documentType': parser.doc_type,
            'contentLength': len(result),
            'options': {
                'tableExtraction': enable_table_extraction,
                'ocr': enable_ocr,
                'markdownFormat': is_md_response
            }
        }
        
        # Upload to S3 if bucket is configured
        if output_bucket:
            # Generate output key
            filename = extract_filename_from_url(presigned_url)
            timestamp = datetime.utcnow().strftime('%Y/%m/%d')
            unique_id = str(uuid.uuid4())[:8]
            output_key = f"{output_prefix}/{timestamp}/{unique_id}_{filename}.md"
            
            # Upload markdown to S3
            s3_uri = upload_to_s3(result, output_bucket, output_key)
            logger.info(f"Uploaded markdown to: {s3_uri}")
            
            response_data['s3Uri'] = s3_uri
            response_data['bucket'] = output_bucket
            response_data['key'] = output_key
            
            # Generate presigned URL for viewing
            if generate_url:
                view_url = generate_presigned_url(output_bucket, output_key, url_expiration)
                response_data['viewUrl'] = view_url
                logger.info(f"Generated presigned URL for viewing (expires in {url_expiration}s)")
        else:
            # Return markdown content directly if no S3 bucket configured
            response_data['markdown'] = result
        
        return {
            'statusCode': 200,
            'body': json.dumps(response_data)
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {
            'statusCode': 502,
            'body': json.dumps({'error': f'Error fetching document: {str(e)}'})
        }
    except Exception as e:
        # Get full stack trace for debugging
        stack_trace = traceback.format_exc()
        logger.error(f"Error processing document: {str(e)}\nStack trace: {stack_trace}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Processing error: {str(e)}'})
        }
