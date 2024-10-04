# main.py
"""
Main file for the classic model architecture project.

This script serves as the entry point of the project. It orchestrates the entire workflow,
including data preparation, model training, prediction, and evaluation.

Key Responsibilities:
- Load configurations and required dependencies.
- Manage the flow by invoking functions from other modules (data preparation, model, train, predict, evaluate).
- Orchestrate the entire process from data preparation to model evaluation.

File Structure:
- images/: Directory containing all image files for the project.
- models/: Directory where trained models will be saved.
- classified_tool_names.txt: Contains the name and label for each image in the format:
  'image_file_name: label'.

Key Considerations:
- Make use of GPU (RTX 4080) for faster training.
- Add support for using mixed precision training to improve performance and memory usage on the GPU.
"""
import tensorflow as tf
from data_preparation import load_and_augment_data, split_data, preprocess_images
from model import build_model, compile_model
from train import train_model
from predict import make_predictions
from evaluate import evaluate_model, save_evaluation_results

def main():
    # Enable mixed precision training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Load and augment data
    images, labels, unique_labels, label_to_index = load_and_augment_data(augmentation_factor=5)
    
    # Split the data
    train_data, val_data, test_data = split_data(images, labels)
    
    # Preprocess the data
    train_data, val_data, test_data = preprocess_images(train_data, val_data, test_data)
    
    num_classes = len(unique_labels)
    
    model = build_model(num_classes=num_classes)
    compile_model(model)
    train_model(model, train_data, val_data)

    # Model prediction
    make_predictions(model)

    # Model evaluation
    cm, unique_labels = evaluate_model(model, test_data, label_to_index)  # Pass the label_to_index dictionary to evaluate_model
    save_evaluation_results(cm, unique_labels)

if __name__ == "__main__":
    main()