from flask import Flask, request, jsonify
import cv2
import pytesseract
from flask_swagger_ui import get_swaggerui_blueprint
import numpy as np
from utils.image_processing import load_image, preprocess_image
from utils.text_extraction import extract_text, clean_text
from utils.document_classification import classify_document_fuzzy
from logger import Logger
# Initialize the Flask app

app = Flask(__name__)

SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger.json'  # Our API url (can of course be a local resource)

API_KEY = "thisismyapikey"

def validate_api_key():
    # Validate the API key from the request headers.
    api_key = request.headers.get('X-API-KEY')
    if api_key is None or api_key != API_KEY:
        return False
    return True

logger = Logger()
# Middleware: before each request
@app.before_request
def before_request_func():
    # This will run before every request
    logger.info(f"Request from {request.remote_addr} at {request.method} {request.url}")

# Route for document classification
@app.route('/api/scanner', methods=['POST'])
def classify_document():
    # Validate the API key
    if not validate_api_key():
        return jsonify({"message": "Unauthorized access"}), 401
    try:
        # Get the image file from the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the image file
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Preprocess the image
        gray, mask = preprocess_image(image)

        # Extract text from the preprocessed mask
        text = extract_text(mask)
        if text == "no text":
            return jsonify({'warning':'Blank Document detected'})
        elif text == "few text":
            return jsonify({'error':'Please enter a clearer image'})
        else:
            cleaned_text = clean_text(text)

        # Classify the document
        classification_result, confidence = classify_document_fuzzy(cleaned_text)

        # Return the classification result as a JSON response
        return jsonify({
            'extracted_text': cleaned_text,
            'classification': classification_result,
            'confidence': round(confidence * 100, 2)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    },
    # oauth_config={  # OAuth config. See https://github.com/swagger-api/swagger-ui#oauth2-configuration .
    #    'clientId': "your-client-id",
    #    'clientSecret': "your-client-secret-if-required",
    #    'realm': "your-realms",
    #    'appName': "your-app-name",
    #    'scopeSeparator': " ",
    #    'additionalQueryStringParams': {'test': "hello"}
    # }
)

app.register_blueprint(swaggerui_blueprint)

# Main entry point
if __name__ == '__main__':
    port = 8000
    app.run(debug=True,port=port)
    logger.info(f"Server started and is now listening on port: {port}")
