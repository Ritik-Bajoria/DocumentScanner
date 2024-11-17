import pytesseract
import cv2
import re
import sys
import os 

## Constants
if os.path.exists("/.dockerenv"):
    TESSERACT_PATH = r'../usr/bin/tesseract' # FOR RUNNING IN DOCKER
    custom_config = r'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata/"' # FOR RUNNING IN DOCKER
else:
    TESSERACT_PATH =r'C:/Program Files/Tesseract-OCR/tesseract.exe' # IF RUNNING IN LOCAL

# Initialize Tesseract
# Perform OCR and pass the tessdata directory in the config
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text(mask):
    #Use pytesseract to extract text from the binary mask.#
    if os.path.exists("/.dockerenv"):
        text = pytesseract.image_to_string(mask, lang = 'nep+eng', config = custom_config ) # FOR RUNNING IN DOCKER
    else:
        text = pytesseract.image_to_string(mask, lang = 'nep+eng') # FOR RUNNING IN LOCAL AND COMMENT ABOVE LINE

    if(len(text)<=5):
        return "no text"
    elif(len(text)<=100):
        return "few text"
    else:
        return text

def clean_text(text):
    # Clean the extracted text by removing unwanted characters while keeping Nepali, English, numbers, spaces, colons, and newlines.
    cleaned_text = re.sub(r'[^A-Za-z0-9\u0900-\u097F\s:\n]', '', text)  # Keep English, Nepali (Devanagari), numbers, spaces, colons, and newlines
    cleaned_text = re.sub(r'(\n\s*)+', '\n', cleaned_text)  # Normalize multiple newlines to a single newline
    return cleaned_text
