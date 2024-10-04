# data_preparation.py
"""
Data preparation for the classic model architecture project.

This script handles the preprocessing of the dataset, which includes:
- Loading image files and corresponding labels from 'images/' and 'classified_tool_names.txt'.
- Splitting the dataset into training, validation, and test sets.
- Performing necessary transformations like resizing, normalizing, and augmenting images if required.

File Structure:
- images/: Directory containing all project images.
- classified_tool_names.txt: Contains the mapping of image file names to their labels in the format:
  'image_file_name: label'.

Enhancements for Performance:
- Apply image augmentation techniques like random rotations, flips, zooms, and brightness adjustments to make the model more robust.
- Normalize pixel values to a range of [0, 1].
- Consider using efficient data loading and preprocessing techniques such as TensorFlow’s `tf.data` API for seamless data pipeline preparation.

Example:
- 'wireline_slickline_tools_mechanical_spang_jar_qls_1_25inch_2_5inch_6.jpg: Jars (Hydraulic or Mechanical)'
- 'api_pear_drop_rope_socket_wireline_tool_string_variable_connection_type.jpg: Joints and Connectors'
"""
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_augment_data(augmentation_factor=10):
    images_dir = 'Luna-ImageDetection/images/'
    labels_file = 'Luna-ImageDetection/classified_tool_names.txt'
    
    # Load image file names and labels
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    file_labels = [line.strip().split(': ') for line in lines]
    
    # Create an ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,           # Rotación de hasta 40 grados
        width_shift_range=0.2,       # Desplazamiento horizontal del 20%
        height_shift_range=0.2,      # Desplazamiento vertical del 20%
        shear_range=0.2,             # Deformación
        zoom_range=[0.8, 1.2],       # Zoom de entre 80% y 120%
        horizontal_flip=True,        # Voltear horizontalmente
        vertical_flip=True,          # Voltear verticalmente
        brightness_range=[0.8, 1.2], # Variación en brillo
        channel_shift_range=30.0,    # Desplazamiento de canales de color
        fill_mode='reflect'          # Cómo rellenar los píxeles vacíos tras la transformación
    )

    # Load and augment images and labels
    images = []
    labels = []
    for file_name, label in file_labels:
        image_path = os.path.join(images_dir, file_name)
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')  # Ensure image is in RGB format
                img = img.resize((224, 224))  # Resize image
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
                
                # Add the original image
                images.append(img_array)
                labels.append(label)
                
                # Generate augmented images
                img_array = np.expand_dims(img_array, 0)
                augmented_images = datagen.flow(img_array, batch_size=1)
                for i in range(augmentation_factor - 1):  # -1 because we already added the original
                    aug_img = next(augmented_images)[0]
                    images.append(aug_img)
                    labels.append(label)
        except Exception as e:
            print(f"Skipping file due to error: {image_path}")
            print(f"Error message: {str(e)}")
            continue
    
    unique_labels = sorted(set(labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    
    numeric_labels = [label_to_index[label] for label in labels]
    
    print(f"Number of unique labels: {len(unique_labels)}")
    print("Label mappings:")
    for label, index in label_to_index.items():
        print(f"{label}: {index}")
    
    print(f"Total images after augmentation: {len(images)}")
    print(f"Total labels after augmentation: {len(numeric_labels)}")
    
    return np.array(images), np.array(numeric_labels), unique_labels, label_to_index

def split_data(images, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0
    
    dataset_size = len(images)
    indices = np.random.permutation(dataset_size)
    
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_images, train_labels = images[train_indices], labels[train_indices]
    val_images, val_labels = images[val_indices], labels[val_indices]
    test_images, test_labels = images[test_indices], labels[test_indices]
    
    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_data = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    
    return train_data, val_data, test_data

def preprocess_images(train_data=None, val_data=None, test_data=None):
    def normalize(image, label):
        return image, label
    
    if train_data is not None:
        train_data = train_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    if val_data is not None:  
        val_data = val_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    if test_data is not None:
        test_data = test_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_data, val_data, test_data