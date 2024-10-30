import cv2
import numpy as np

THRESHOLD_VALUE = 110

def load_image(image_path):
    # Load an image from a specified path.# 
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")
    return image

# def preprocess_image(image):
#     # Convert the image to grayscale and create a binary mask.#
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
#     return gray, mask

def preprocess_image(image):
 
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gaussian Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Sharpen the image using a kernel
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)
    
    # 4. Apply adaptive thresholding for better binarization in varying lighting
    # _, mask = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 30)
    
    # mask = correct_orientation(mask)
    
    return gray, mask

# def correct_orientation(mask):
#     # Invert the mask for contour detection if needed (white text on black background)
#     inverted_mask = cv2.bitwise_not(mask)

#     # Find contours of the text areas
#     contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # If no contours found, return the original mask
#     if not contours:
#         return mask

#     # Get the minimum bounding box around the largest contour, assuming it's the main text area
#     largest_contour = max(contours, key=cv2.contourArea)
#     rect = cv2.minAreaRect(largest_contour)
#     angle = rect[-1]

#     # Adjust angle based on the rectangular orientation
#     if angle < -45:
#         angle += 90  # Adjust angle to be within [-45, 45]

#     # Print the angle for debugging purposes
#     print(f"Detected angle: {angle}")

#     # Check if the angle is within a stricter range close to horizontal alignment
#     if -5 <= angle <= 5:  # Adjusted to a tighter threshold
#         return mask  # No rotation needed

#     # Get the rotation matrix to deskew
#     (h, w) = mask.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)

#     # Rotate the image to correct orientation
#     corrected_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return corrected_mask