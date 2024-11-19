import os
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory

Tk().withdraw()

# Load the Tesseract-OCR engine
def ask_image_folder():
    # for dynamic path
    image_folder = askdirectory(title="Select Folder Containing Images  to train the model")

    # Validate the selected path
    if image_folder:
        if os.path.exists(image_folder):
            print(f"Images will be loaded from: {image_folder}")
            return image_folder
        else:
            print("The selected path is invalid.")
            sys.exit(0)
    else:
        print("No folder selected.")
        sys.exit(0)

def load_images(image_folder):
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff') 
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((img, filename))
    return images 

def resolution_detection(image, min_height = 300, min_width = 300):
    # Detect the resolution of the image

    # Get the dimensions of the image
    height, width = image.shape[:2]
    print(f"Image resolution: {width}x{height}")
    
    # Check if the resolution meets the threshold
    if width >= min_width and height >= min_height:
        return True # image has good resolution
    else:
        return False # image does not have good resolution

def sharpness_detection(image, threshold = 200):
    # Detect the sharpness of the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    # classify the image as sharp or not sharp as per the threshold
    if blur_score > threshold:
        return True # image is sharp
    else:
        return False # image is not sharp


def noise_detection(image, threshold = 150):
    # Detect the noise level of the image
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a sliding window (kernel) to calculate the variance
    ksize = 5  # Kernel size for local variance
    image_height, image_width = image.shape
    
    # Create a copy of the image for the result
    variance_map = np.zeros((image_height, image_width), dtype=np.float32)
    
    # Loop through the image in a sliding window fashion
    for i in range(ksize//2, image_height-ksize//2):
        for j in range(ksize//2, image_width-ksize//2):
            # Extract the local region
            patch = image[i-ksize//2:i+ksize//2+1, j-ksize//2:j+ksize//2+1]
            # Compute the variance of pixel intensities in the patch
            patch_variance = np.var(patch)
            variance_map[i, j] = patch_variance
    
    # Calculate the overall image variance
    overall_variance = np.mean(variance_map)
    print(overall_variance)
    # If overall variance is lower than the threshold, consider the image clean (not noisy)
    if overall_variance < threshold:
        return True  # Image is clean (not noisy)
    else:
        return False  # Image is noisy

def contrast_detection(image, threshold=40):
    # Detect the contrast of the image
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the standard deviation of pixel intensities
    contrast_value = np.std(gray)
    print(contrast_value)
    # Determine if the contrast is acceptable
    if contrast_value >= threshold:
        return True  # High contrast
    else:
        return False  # Low contrast

def distinguish_image_quality(images,image_folder):
    # This function will be used to identify the quality of the images in the selected folder
    for image, image_name in images:
        # Quality identification 
        is_high_resolution = resolution_detection(image)
        is_sharp = sharpness_detection(image)
        is_not_noisy = noise_detection(image)
        # has_high_OCR_confidence = OCR_detection(image)
        # Ensure the directories exist
        os.makedirs("high_quality_images", exist_ok=True)
        os.makedirs("low_quality_images", exist_ok=True)
        has_high_contrast = contrast_detection(image)
        print(f"\n\n{image_name}::\n is high resolution : {is_high_resolution}\nis sharp : {is_sharp}\nis_not_noisy : {is_not_noisy}\nhas high contrast : {has_high_contrast}")
        if is_high_resolution and is_sharp and is_not_noisy and has_high_contrast: 
            path = f"high_quality_images/{image_name}"
            cv2.imwrite(path,image)
        else:
            path = f"low_quality_images/{image_name}"
            cv2.imwrite(path,image)

if __name__ == '__main__':
    image_folder = ask_image_folder()
    images = load_images(image_folder)
    distinguish_image_quality(images,image_folder)