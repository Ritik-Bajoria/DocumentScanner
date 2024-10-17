import cv2
import re
import pytesseract
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Constants
TESSERACT_PATH = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
THRESHOLD_VALUE = 122

# Initialize Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def load_image(image_path):
    """Load an image from a specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")
    return image

def preprocess_image(image):
    """Convert the image to grayscale and create a binary mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    return gray, mask

def extract_text(mask):
    """Use pytesseract to extract text from the binary mask."""
    return pytesseract.image_to_string(mask)

def clean_text(text):
    """Clean the extracted text by removing unwanted characters."""
    cleaned_text = re.sub(r'[^A-Za-z0-9\s:\n]', '', text)  # Keep alphanumeric, spaces, colons, and newlines
    cleaned_text = re.sub(r'(\n\s*)+', '\n', cleaned_text)  # Normalize multiple newlines to a single newline
    return cleaned_text.strip()  # Remove any leading or trailing whitespace

def fuzzy_membership_score(cleaned_text, keyword_pairs):
    """
    Calculate a fuzzy membership score based on the presence of keywords.
    Keywords are provided as pairs (Nepali, English), and the score is computed based on the presence of either keyword.
    """
    score = sum(1 for nepali, english in keyword_pairs if nepali in cleaned_text or english in cleaned_text)
    return score / len(keyword_pairs) if keyword_pairs else 0.0

def classify_document_fuzzy(cleaned_text):
    """Classify the document using fuzzy logic based on keyword presence."""
    # Define keyword sets for different document types
    # List of keywords and phrases commonly found in a Nepali Citizenship Certificate
    citizenship_keywords = [
        ("गृह मन्त्रालय", "Ministry of Home Affairs"),
        ("नेपाल सरकार", "Government of Nepal"),
        ("जन्म मिति", "Date of Birth"),
        ("नागरिकताको प्रकार", "Type of Citizenship"),
        ("स्थायी ठेगाना", "Permanent Address"),
        ("नाम", "Name"),
        ("बाबुको नाम", "Father's Name"),
        ("आमाको नाम", "Mother's Name"),
        ("पति/पत्नीको नाम", "Spouse's Name"),
        ("नागरिकता नम्बर", "Citizenship Number"),
        ("दर्ता मिति", "Date of Registration"),
        ("फोटो", "Photograph")
    ]

    # List of keywords and phrases commonly found in a PAN Card
    pan_card_keywords = [
        ("नेपाल सरकार", "Government of Nepal"),
        ("आन्तरिक राजस्व विभाग", "Inland Revenue Department"),
        ("Permanent Account Number", "Permanent Account Number"),
        ("कार्डधारीको नाम", "Cardholder's Name"),
        ("जन्म मिति", "Date of Birth"),
        ("लिङ्ग", "Gender"),
        ("ठेगाना", "Address"),
        ("जारी मिति", "Date of Issue"),
        ("फोटो", "Photograph")
    ]

    # List of keywords and phrases commonly found in a Passport
    passport_keywords = [
        ("नेपाल सरकार", "Government of Nepal"),
        ("विदेश मन्त्रालय", "Ministry of Foreign Affairs"),
        ("Passport Number", "Passport Number"),
        ("नाम", "Name"),
        ("लिङ्ग", "Gender"),
        ("राष्ट्रियता", "Nationality"),
        ("जन्म मिति", "Date of Birth"),
        ("जन्मस्थान", "Place of Birth"),
        ("पेसा", "Profession"),
        ("स्थायी ठेगाना", "Permanent Address"),
        ("पासपोर्ट जारी मिति", "Passport Issue Date"),
        ("पासपोर्ट समाप्त मिति", "Passport Expiry Date"),
        ("जारी गर्ने प्राधिकरण", "Issuing Authority"),
        ("फोटो", "Photograph"),
        ("हस्ताक्षर", "Signature")
    ]

    # Calculate fuzzy membership scores
    citizenship_score = fuzzy_membership_score(cleaned_text, citizenship_keywords)
    pan_card_score = fuzzy_membership_score(cleaned_text, pan_card_keywords)
    passport_score = fuzzy_membership_score(cleaned_text, passport_keywords)

    # Determine the classification based on the highest score
    classification = "Unknown Document"
    confidence = 0.0
    print(pan_card_score)
    print(passport_score)
    print(citizenship_score)
    if citizenship_score or pan_card_score or passport_score > 0.4:
        if citizenship_score > pan_card_score and citizenship_score > passport_score:
            classification = "Nepali Citizenship Document Detected"
            confidence = citizenship_score
        elif pan_card_score > citizenship_score and pan_card_score > passport_score:
            classification = "PAN Card Detected"
            confidence = pan_card_score
        elif passport_score > citizenship_score and passport_score > pan_card_score:
            classification = "Passport Detected"
            confidence = passport_score
    else:
        confidence  = "unknown document"


    return classification, confidence

def main():
    """Main function to execute the workflow."""
    try:
        # Set up tkinter root
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Open a file dialog to select an image
        image_path = filedialog.askopenfilename(title="Select an Image", 
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not image_path:  # Check if the user canceled the dialog
            print("No image selected.")
            return
        
        # Load and display the original document
        image = load_image(image_path)
        cv2.imshow('Original Document', image)
        
        cv2.waitKey(0)

        # Pre-process the image
        gray, mask = preprocess_image(image)
        cv2.imshow('Grayscaled Document', gray)
        cv2.waitKey(0)
        cv2.imshow('Threshold Binary Document', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Extract text from the mask
        text = extract_text(mask)
        cleaned_text = clean_text(text)

        # Classify the document using fuzzy logic and print results
        classification_result, confidence = classify_document_fuzzy(cleaned_text)
        print(f"Extracted Text:\n{cleaned_text}")
        print(f"Classification Result: {classification_result}")
        confidence_level = confidence * 100
        print(f"Confidence Level in %: {confidence_level:.2f}")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
