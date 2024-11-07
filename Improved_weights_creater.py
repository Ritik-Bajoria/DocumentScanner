import os
import numpy as np
import cv2
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def build_espcn_model(scale_factor):
    model = models.Sequential()
    model.add(layers.Input(shape=(None, None, 3)))
    
    # Convolution layers with increased depth
    model.add(layers.Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    
    # Output layer with pixel shuffle
    model.add(layers.Conv2D(3 * (scale_factor ** 2), (3, 3), padding='same'))
    model.add(layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor)))
    
    return model

def load_images(image_folder, target_size=(255, 255)):
    images = []
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            images.append(img)
    return np.array(images, dtype='float32') / 255.0

def create_low_res_images(high_res_images, scale_factor):
    low_res_images = []
    for img in high_res_images:
        low_res_img = cv2.GaussianBlur(img, (5, 5), 0)
        low_res_img = cv2.resize(low_res_img, (img.shape[1] // scale_factor, img.shape[0] // scale_factor), interpolation=cv2.INTER_AREA)
        low_res_images.append(low_res_img)
    return np.array(low_res_images)

def preprocess_images(images):
    return images.astype('float32') / 255.0

# Load and preprocess images
image_folder = "C:\\Users\\Legion\\Downloads\\archive\\DIV2K_valid_HR\\DIV2K_valid_HR"
high_res_images = load_images(image_folder, target_size=(255, 255))
scale_factor = 3
low_res_images = create_low_res_images(high_res_images, scale_factor)

# Data augmentation for robustness
def augment_images(images):
    augmented_images = []
    for img in images:
        augmented_images.append(cv2.flip(img, 1))  # Horizontal flip
    return np.concatenate([images, np.array(augmented_images)])

# Apply augmentation
low_res_images = augment_images(preprocess_images(low_res_images))
high_res_images = augment_images(preprocess_images(high_res_images))

# Build and compile the model
model = build_espcn_model(scale_factor)
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# Callbacks
name = f'div2k_espcn_weights_x{scale_factor}_{datetime.now().strftime("%Y-%m-%d")}.weights.h5'
callbacks = [
    ModelCheckpoint(name, save_best_only=True, monitor='loss', mode='min', verbose=1, save_weights_only=True),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
    EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
]

# Train the model
model.fit(
    low_res_images, 
    high_res_images, 
    epochs=50, 
    batch_size=16, 
    callbacks=callbacks, 
    validation_split=0.1
)

print(f"Training complete. Weights saved as {name}.")
