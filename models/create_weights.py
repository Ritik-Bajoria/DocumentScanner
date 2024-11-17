import os
import sys
import numpy as np
import cv2
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tkinter import Tk
from tkinter.filedialog import askdirectory

# Hide the root Tkinter window
Tk().withdraw()
def build_espcn_model(scale_factor):
    model = models.Sequential()
    
    # Define input layer separately to avoid warnings
    model.add(layers.Input(shape=(None, None, 3)))
    
    # First convolution layer
    model.add(layers.Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    
    # Output layer with required channels for pixel shuffle
    model.add(layers.Conv2D(3 * (scale_factor ** 2), (3, 3), padding='same'))

    # Pixel Shuffle layer using Lambda
    model.add(layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor)))
    
    return model

def load_images(image_folder, target_size=(255, 255)):
    images = []
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Resize to target size
            img = cv2.resize(img, target_size)
            # Convert grayscale images to RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            images.append(img)
    return np.array(images, dtype='float32') / 255.0  # Normalize to [0, 1]

def create_low_res_images(high_res_images, scale_factor):
    low_res_images = []
    for img in high_res_images:
        # Resize using a Gaussian filter for downsampling
        low_res_img = cv2.GaussianBlur(img, (5, 5), 0)
        low_res_img = cv2.resize(low_res_img, (img.shape[1] // scale_factor, img.shape[0] // scale_factor), interpolation=cv2.INTER_AREA)
        low_res_images.append(low_res_img)
    return np.array(low_res_images)

def preprocess_images(images):
    images = images.astype('float32') / 255.0  # Normalize images
    return images

# Load high-resolution images
# for static path
# image_folder = "C:\\Users\\Legion\\Downloads\\archive\\DIV2K_valid_HR\\DIV2K_valid_HR"  # Replace with your folder path

# for dynamic path
image_folder = askdirectory(title="Select Folder Containing Images  to train the model")

# Validate the selected path
if image_folder:
    if os.path.exists(image_folder):
        print(f"Images will be loaded from: {image_folder}")
    else:
        print("The selected path is invalid.")
        sys.exit(0)
else:
    print("No folder selected.")
    sys.exit(0)

# load the images from the path
high_res_images = load_images(image_folder, target_size=(255, 255))

# Create low-resolution images
scale_factor = 3  # Set the desired scale factor
low_res_images = create_low_res_images(high_res_images, scale_factor)

# Preprocess images
low_res_images = preprocess_images(low_res_images)
high_res_images = preprocess_images(high_res_images)

# Build the model
model = build_espcn_model(scale_factor)

# Compile the model
model.compile(optimizer=Adam(), loss=MeanSquaredError())

# Train the model
model.fit(low_res_images, high_res_images, epochs=50, batch_size=16)

name = f'div2k_espcn_weights_x{scale_factor}_{datetime.now().strftime('%Y-%m-%d')}.weights.h5'
# Save the model weights
model.save_weights(f"{name}")  # Save the trained weights

print(f"Training complete and weights saved as {name}.")
