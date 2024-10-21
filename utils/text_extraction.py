import pytesseract
import cv2
import re

# Constants
TESSERACT_PATH = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
# Initialize Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text(mask):
    #Use pytesseract to extract text from the binary mask.#
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
