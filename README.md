# Document Scanner API

## Overview
This project provides a Flask-based API for a Document Scanner application. It enables users to upload images of documents, preprocess them, extract text using OCR, classify the documents, and return structured information. It includes Swagger UI for API documentation and logging functionality.

## Features
- Document classification for two API versions.
- Preprocessing of uploaded images to improve OCR accuracy.
- Text extraction and cleaning using Tesseract OCR.
- Document classification with confidence levels.
- Graceful shutdown with signal handlers.
- Validation of API requests via API keys.
- Swagger UI for API documentation.
- Middleware for request logging.

## Prerequisites
1. Python 3.6+
2. Docker (optional for deployment)
3. Tesseract OCR installed
4. Environment variables:
   - `API_KEY`: API key for authentication.
   - `PORT`: Port to run the application.
   - `HOST`: Host address for the application.

### Swagger Documentation
- URL: `/api/docs`
- Description: Interactive documentation for the API.

### Scanner API
#### Endpoint:
`POST /api/v<int:version>/scanner`

#### Headers:
- `X-API-KEY`: API key for authentication.

#### Body:
- File upload with the key `file`.

## Notes
- Ensure Tesseract OCR is installed and accessible from the environment where the application runs.
- Adjust the confidence threshold and preprocessing as needed for better accuracy.
