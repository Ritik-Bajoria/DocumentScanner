import sys
import cv2
import re
import pytesseract
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Constants
TESSERACT_PATH = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Initialize Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def load_image(image_path):
    # Load an image from a specified path.# 
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")
    return image

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply denoising to reduce noise
    denoised = cv2.fastNlMeansDenoising(gray,None, 11, 7, 13)
    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(denoised, (7, 5), 0)
    cv2.imshow('blurred',blurred)
    cv2.waitKey(0)
    # Apply adaptive thresholding
    mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 13, 2)
    return gray, mask

def extract_text(mask):
    #Use pytesseract to extract text from the binary mask.#
    osd = pytesseract.image_to_osd(mask, output_type='dict')
    print(osd)
    text = pytesseract.image_to_string(mask, lang = 'nep+eng')
    # Get the image dimensions
    (h, w) = mask.shape[:2]

    # Define the rotation angle
    angle = -90  # Rotate -90 degrees

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Generate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    for i in range(3):
        print("text length is",len(text))
        if(len(text)<=100):
            mask = cv2.warpAffine(mask, M, (w, h))
            cv2.imshow("rotated mask",mask)
            cv2.waitKey(0)
            text = pytesseract.image_to_string(mask, lang = 'nep+eng')
            text = clean_text(text)
        else:
            break
        angle += 90
    
    if(len(text)<=100):
        print("\nPlease insert a clearer image\n")
        sys.exit()
    else:
        return text

def clean_text(text):
    # Clean the extracted text by removing unwanted characters while keeping Nepali, English, numbers, spaces, colons, and newlines.#
    cleaned_text = re.sub(r'[^A-Za-z0-9\[u0900-\u097F\s:\n]', '', text)  # Keep English, Nepali (Devanagari), numbers, spaces, colons, and newlines
    cleaned_text = re.sub(r'(\n\s*)+', '\n', cleaned_text)  # Normalize multiple newlines to a single 
    return cleaned_text.strip()  # Remove any leading or trailing whitespace

# def convert_words_to_char_sets(words):
#     """Convert a list of words into a list of sets of characters."""
#     return [set(word) for word in words]

def match_words(word, cleaned_text):
    # for text in cleaned_text:
    #     matched_characters = convert_words_to_char_sets(word).intersection(convert_words_to_char_sets(text))
    #     similarity_ratio = len(matched_characters) / min(len(word), len(text))
    #     if similarity_ratio > 0.7:
    #         break

    # if similarity_ratio > 0.7:
    #     return true
    # else:
    #     return false
    if word in cleaned_text:
        return True
    else:
        return False

def fuzzy_membership_score(cleaned_text, keyword_pairs):
    """
    Calculate a fuzzy membership score based on the presence of keywords.
    Keywords are provided as pairs (Nepali, English), and the score is computed based on the presence of either keyword.
    """
    score = sum(1 for nepali, english in keyword_pairs if (match_words(nepali,cleaned_text)) or (match_words(english,cleaned_text)))
    return score / len(keyword_pairs) if keyword_pairs else 0.0

def classify_document_fuzzy(cleaned_text):
    # Classify the document using fuzzy logic based on keyword presence.#
    # Define keyword sets for different document types
    # List of keywords and phrases commonly found in a Nepali Citizenship Certificate
    citizenship_keywords = [
        ("नेपाल सरकार", "Government of Nepal"),
        ("गृह मन्त्रालय", "Ministry of Home Affairs"),
        ("जन्म मिति", "Date of Birth"),
        ("नागरिकताको प्रकार", "Type of Citizenship"),
        ("स्थायी ठेगाना", "Permanent Address"),
        ("नाम", "Name"),
        ("बाबुको नाम", "Father's Name"),
        ("आमाको नाम", "Mother's Name"),
        ("लिङ्ग", "Gender"),
        ("नेपाली नागरिकता प्रमाणपत्र", "Nepali Citizenship Certificate"),
        ("ठेगाना", "Address"),
        ("जिल्ला", "District")
    ]

    # List of keywords and phrases commonly found in a PAN Card
    pan_card_keywords = [
        ("नेपाल सरकार", "Government of Nepal"),
        ("आन्तरिक राजस्व विभाग", "Inland Revenue Department"),
        ("Permanent Account Number", "Permanent Account Number"),
        ("नाम", "Name"),
        ("जन्म मिति", "Date of Birth"),
        ("स्थायी.ले.न.", "PAN"),
        ("ठेगाना", "Address"),
        ("मिति", "Date"),
        ("सही", "Signature")
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
    if citizenship_score > 0.4 or pan_card_score > 0.4 or passport_score > 0.4:
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
        classification  = "unknown document"

    return classification, confidence

#main function controlling overall workflow000000
def main():
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
        text = clean_text(text)
        # Classify the document using fuzzy logic and print results
        classification_result, confidence = classify_document_fuzzy(text)
        print(f"Extracted Text:\n{text}")
        print(f"Classification Result: {classification_result}")
        print(confidence)
        confidence_level = confidence * 100
        print(f"Confidence Level in %: {confidence_level:.2f}")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
