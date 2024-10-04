# predict.py
"""
Prediction script for the classic model architecture project.

This script is responsible for loading a trained model and making predictions on new, unseen images. 
It will also provide functionality to save the prediction results.

Key Responsibilities:
- Load a trained model from the 'models/' directory.
- Make predictions on images from 'images/'.
- Output the predicted labels.

File Structure:
- models/: Directory from where the trained model will be loaded.

Enhancements:
- Use efficient batching for predictions to optimize GPU utilization.
- Consider saving predictions in a CSV or JSON format for easier analysis.
"""
import tensorflow as tf
import os
from PIL import Image
import numpy as np

def load_model(model_path='models/classic_model_best.keras'):
    model = tf.keras.models.load_model(model_path)
    return model

def make_predictions(model, batch_size=32):
    images_dir = 'images/'
    image_files = os.listdir(images_dir)
    
    images = []
    valid_file_names = []
    for file_name in image_files:
        image_path = os.path.join(images_dir, file_name)
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')  # Convert to RGB format
            image = image.resize((224, 224))  # Resize to match model input size
            image = np.array(image) / 255.0  # Normalize pixel values
            images.append(image)
            valid_file_names.append(file_name)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue
    
    if not images:
        print("No valid images found for prediction.")
        return

    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    predictions = model.predict(dataset)
    predicted_labels = tf.argmax(predictions, axis=1)
    
    # Save predictions to a file
    with open('predictions.txt', 'w') as f:
        for file_name, label in zip(valid_file_names, predicted_labels):
            f.write(f"{file_name}: {label}\n")

