import cv2

THRESHOLD_VALUE = 122

def load_image(image_path):
    # Load an image from a specified path.# 
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")
    return image

def preprocess_image(image):
    # Convert the image to grayscale and create a binary mask.#
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    return gray, mask