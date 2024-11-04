import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

def load_image(image_path):
    # Load an image from a specified path.# 
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")
    image = image.astype('float32')/255.0
    image = apply_super_resolution(model,image)
    image = image *255.0
    image = np.clip(image, 0, 255)  # Ensure pixel values are within [0, 255]
    image = image.astype(np.uint8)  # Convert to uint8 for image representation
    return image

def preprocess_image(image):
 
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gaussian Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Sharpen the image using a kernel
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)
    
    # 4. Apply adaptive thresholding for better binarization in varying lighting
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 30)
    
    return gray, mask

def build_espcn_model(scale_factor):
    model = models.Sequential()

    # First convolution layer
    model.add(layers.Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=(None, None, 3)))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    
    # Output layer
    model.add(layers.Conv2D(3 * (scale_factor ** 2), (3, 3), padding='same'))

    # Pixel Shuffle layer using Lambda
    model.add(layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor)))
    
    return model

# Load the pre-trained model
model = build_espcn_model(scale_factor=2)

# Load pre-trained weights (Make sure you have this file)
model.load_weights('ESPCN-x4.h5')  # Replace with your actual weights file path

def apply_super_resolution(model, low_res_image):
    # Expand dimensions to include batch size
    low_res_image = np.expand_dims(low_res_image, axis=0)  # Shape: (1, height, width, 3)
    
    # Predict the super-resolved image
    sr_image = model.predict(low_res_image)[0]  # Take the first (and only) element
    sr_image = np.clip(sr_image, 0, 1)  # Ensure pixel values are between 0 and 1
    return sr_image