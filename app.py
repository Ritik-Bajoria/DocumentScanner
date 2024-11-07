from flask import Flask, request, jsonify
import cv2
import pytesseract
from flask_swagger_ui import get_swaggerui_blueprint
import numpy as np
from utils.image_processing import preprocess_image
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
        # Save gray and mask images in the same folder
        cv2.imwrite('gray_image.png', gray)

        # Get the image dimensions
        (h, w) = mask.shape[:2]

        # Define the rotation angle
        angle = 0  

        # Calculate the center of the image
        center = (w // 2, h // 2)

        i = 0
        text =""
        confidence=0
        while True:
            print("text length is",len(text))
            if(confidence<=0.4):
                # Generate the rotation matrix
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                mask = cv2.warpAffine(mask, M, (w, h))
                # Extract text from the preprocessed mask
                text = extract_text(mask)
                # Classify the document
                text = clean_text(text)
                classification_result, confidence = classify_document_fuzzy(text)
            else:
                break
            cv2.imwrite('mask_image.png', mask)
            angle += 90
            if(i>3):
                break
            i+=1
        
        if text == "no text":
            return jsonify({'warning':'Blank Document detected'}), 404
        elif text == "few text":
            return jsonify({'error':'Please enter a clearer image'}), 404

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