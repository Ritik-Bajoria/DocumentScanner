import pytesseract
import cv2
import re
import sys
import os 

## Constants
# TESSERACT_PATH = r'../usr/bin/tesseract' # TESSERACT_PATH ONLY FOR RUNNING IN DOCKER
TESSERACT_PATH =r'C:/Program Files/Tesseract-OCR/tesseract.exe' # UNCOMMENT IF RUNNING IN LOCAL AND COMMENT THE DOCKER TESSERACT_PATH DEFINED ABOVE

# Initialize Tesseract
# Perform OCR and pass the tessdata directory in the config
# custom_config = r'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata/"' #ONLY FOR RUNNING IN DOCKER
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text(mask):
    #Use pytesseract to extract text from the binary mask.#
    # text = pytesseract.image_to_string(mask, lang = 'nep+eng', config = custom_config ) # UNCOMMENT ONLY FOR RUNNING IN DOCKER
    text = pytesseract.image_to_string(mask, lang = 'nep+eng') # UNCOMMENT IF RUNNING IN LOCAL AND COMMENT ABOVE LINE
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
