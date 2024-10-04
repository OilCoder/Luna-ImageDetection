# train.py
"""
Training script for the classic model architecture project.

This script handles the training of the model on the prepared dataset. 
It includes defining the training process, model checkpoints, and saving the best-performing model.

Key Responsibilities:
- Load the prepared dataset from 'data_preparation.py'.
- Train the model using the defined architecture from 'model.py'.
- Save the trained model to the 'models/' directory.

File Structure:
- models/: Directory where trained models will be saved (e.g., 'classic_model_final.keras').

Techniques for Improved Training:
- Use GPU (RTX 4080) for accelerated training.
- Implement model checkpoints to save the best model during training.
- Apply early stopping to avoid overfitting if validation loss doesn't improve.
- Use learning rate scheduling (e.g., ReduceLROnPlateau) to adjust learning rates during training.
- Experiment with batch size to find the optimal number for GPU memory usage.
"""
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import math

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0):
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
        tf.constant(math.pi) *
        (global_step - warmup_steps) / float(total_steps - warmup_steps)))
    if warmup_steps > 0:
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = tf.where(global_step < warmup_steps, warmup_rate, learning_rate)
    return tf.maximum(learning_rate, 1e-6)  # Increased minimum learning rate

def train_model(model, train_data, val_data, epochs=500, batch_size=32):
    steps_per_epoch = len(list(train_data))  # Convert to list to get the length
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * 0.1)  # 10% of total steps for warmup
    
    lr_schedule = LearningRateScheduler(
        lambda epoch: float(cosine_decay_with_warmup(
            epoch * steps_per_epoch,
            learning_rate_base=0.001,
            total_steps=total_steps,
            warmup_steps=warmup_steps
        ))
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('Luna-ImageDetection/models/classic_model_best.keras', 
                                           monitor='val_accuracy', 
                                           save_best_only=True, 
                                           save_weights_only=False),
        lr_schedule,
        reduce_lr,
        tf.keras.callbacks.TerminateOnNaN()
    ]
    
    train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_data = val_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Compute class weights
    class_weights = compute_class_weights(train_data)
    
    model.fit(train_data, 
              epochs=epochs, 
              validation_data=val_data, 
              callbacks=callbacks,
              class_weight=class_weights)
    
    model.save('Luna-ImageDetection/models/classic_model_final.keras')

def compute_class_weights(train_data):
    class_counts = {}
    total_samples = 0
    
    for _, labels in train_data:
        for label in labels:
            label = int(label)
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
            total_samples += 1
    
    class_weights = {label: (total_samples / (len(class_counts) * count)) ** 0.5 for label, count in class_counts.items()}
    return class_weights