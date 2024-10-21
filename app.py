from flask import Flask, request, jsonify
import cv2
import pytesseract
import numpy as np
from utils.image_processing import load_image, preprocess_image
from utils.text_extraction import extract_text, clean_text
from utils.document_classification import classify_document_fuzzy
# Initialize the Flask app

app = Flask(__name__)

# Route for document classification
@app.route('/api/scanner', methods=['POST'])
def classify_document():
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
        if not text:
            return jsonify({'error':'Please enter a clearer image'})
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

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
