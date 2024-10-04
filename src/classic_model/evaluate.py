# evaluate.py
"""
Evaluation script for the classic model architecture project.

This script evaluates the performance of the trained model on a test dataset. 
It includes calculating metrics such as accuracy, precision, recall, and confusion matrix.

Key Responsibilities:
- Load the trained model from 'models/'.
- Evaluate its performance on the test set using standard classification metrics.
- Save evaluation results to the appropriate output file (e.g., 'evaluation_results.png').

File Structure:
- models/: Directory from where the trained model will be loaded.
- evaluation_results.png: File where the evaluation results (e.g., confusion matrix) are saved.

Techniques for Better Evaluation:
- Use a detailed classification report (precision, recall, F1-score) to understand performance across all classes.
- Plot a confusion matrix to visualize model predictions.
- Save evaluation plots (e.g., confusion matrix, ROC curves) to visually assess performance.
"""
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_data, label_to_index, batch_size=32):
    test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    predictions = model.predict(test_data)
    predicted_labels = tf.argmax(predictions, axis=1).numpy()
    true_labels = tf.concat([y for x, y in test_data], axis=0).numpy()
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}") 
    print(f"Recall: {recall:.4f}")
    
    unique_labels = sorted(set(true_labels))
    class_labels = [list(label_to_index.keys())[i] for i in unique_labels]
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, labels=unique_labels, target_names=class_labels, zero_division=0))
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
    
    return cm, class_labels

def save_evaluation_results(cm, unique_labels):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, unique_labels, rotation=45, ha='right')
    plt.yticks(tick_marks, unique_labels)
    
    # Normalize the confusion matrix.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Threshold for text color
    thresh = cm.max() / 2.
    
    # Add text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('evaluation_results_confusion_matrix.png')
    plt.close()
    
    print("Confusion matrix saved as 'evaluation_results_confusion_matrix.png'")