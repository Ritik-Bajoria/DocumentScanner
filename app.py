from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import pytesseract
from flask_swagger_ui import get_swaggerui_blueprint
import numpy as np
from utils.image_processing import preprocess_image1, preprocess_image2
from utils.text_extraction import extract_text, clean_text
from utils.document_classification import classify_document_fuzzy
from logger import Logger
import signal
import sys
# Initialize the Flask app

app = Flask(__name__)
CORS(app)

SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger.json'  # Our API url (can of course be a local resource)

API_KEY = "thisismyapikey"

# Graceful shutdown function
def graceful_shutdown(signal, frame):
    logger.info("Shutting down gracefully...")
    # Perform any cleanup here if needed
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

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
    
# Error response in case of route not found
@app.errorhandler(404)
def not_found_error(e):
    return jsonify({
        "error": True,
        "message": "URL not found"
    }), 404

# Error response in case of method not allowed
@app.errorhandler(405)
def method_not_allowed_error(e):
    return jsonify({
        "error": True,
        "message": "Method not allowed"
    }), 405

# Route for document classification
@app.route('/api/v<int:version>/scanner', methods=['POST'])
def classify_document(version):
    # Validate the API key
    if not validate_api_key():
        return jsonify({
            "error":True,
            "message": "Unauthorized access"
            }), 401
    try:
        # Get the image file from the request
        if 'file' not in request.files:
            return jsonify({
                "error": True,
                'message': 'No file part in the request'
                }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "error": True,
                'message': 'No selected file'
                }), 400

        # Read the image file
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # Preprocess the image
        gray = image
        mask = image
        if version == 1:
            gray, mask = preprocess_image1(image)
        elif version == 2:
            gray, mask = preprocess_image2(image)
        else:
            return jsonify({
                "error": True,
                'message': 'Invalid URL'
                }), 404

        # Save gray and mask images in the same folder
        cv2.imwrite('gray_image.png', gray)

        i = 0
        text =""
        confidence=0
        classification_result = ""
        while True:
            if(confidence<=0.4):
                # Extract text from the preprocessed mask
                extracted_text = extract_text(mask)
                if(len(extracted_text)>=len(text)):
                    text = extracted_text
                    # print("text length is",len(text))
                    # Classify the document
                    text = clean_text(text)
                    classification_result, confidence = classify_document_fuzzy(text)
                else:
                    # print("text length is",len(extracted_text))
                    extracted_text = clean_text(extracted_text)
                    classify_document_fuzzy(extracted_text)

                # Get the image dimensions
                (h, w) = mask.shape[:2]

                # Define the rotation angle
                angle = 90 

                # Calculate the center of the image
                center = (w // 2, h // 2)
                cv2.imwrite('mask_image.png', mask)

                # Generate the rotation matrix
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Calculate the new bounding dimensions of the image
                cos = abs(M[0, 0])
                sin = abs(M[0, 1])

                # Compute new dimensions after rotation
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))

                # Adjust the rotation matrix to consider the translation
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]
                
                mask = cv2.warpAffine(mask, M, (new_w, new_h))
            else:
                break
            if(i>2):
                break
            i+=1

        if text == "no text":
            return jsonify({
                'error':True,
                'warning':'Blank Document detected'
                }), 404
        elif text == "few text":
            return jsonify({
                'error':True,
                'message':'Please enter a clearer image'
                }), 404

        text_to_list = [line.strip() for line in text.strip().split('\n') if line.strip()]  # Split into lines, remove empty lines, and trim whitespace

        # Return the classification result as a JSON response
        # Determine the response code based on classification result
        if classification_result == "Unknown Document":
            return jsonify({
                'extracted_text': text_to_list,
                'classification': classification_result,
                'confidence': round(confidence * 100, 2)
            }), 404
        else:
            return jsonify({
                'extracted_text': text_to_list,
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