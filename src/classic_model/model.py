# model.py
"""
Model definition for the classic model architecture project.

This script defines the model architecture for the classification task. 
The architecture will be a simple, classical CNN or similar architecture suitable for classification.

Key Responsibilities:
- Define the model architecture (e.g., convolutional layers, dense layers, activation functions).
- Compile the model with the appropriate loss function, optimizer, and metrics.

File Structure:
- models/: Directory where trained models will be saved.

Techniques for Improved Performance:
- Use Batch Normalization after each convolution layer to accelerate convergence.
- Add Dropout layers for regularization to prevent overfitting.
- Use advanced optimizers like Adam or RMSprop, and experiment with learning rate schedules.
- Enable mixed precision training with `tensorflow.keras.mixed_precision` for faster computation on GPUs.
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Concatenate, Add, Input
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

policy = mixed_precision.Policy('float32')  # Change to float32 for stability
mixed_precision.set_global_policy(policy)

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    shortcut = x
    if conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same', name=name + '_0_conv')(shortcut)
        shortcut = BatchNormalization(name=name + '_0_bn')(shortcut)

    x = Conv2D(filters, kernel_size, strides=stride, padding='same', name=name + '_1_conv')(x)
    x = BatchNormalization(name=name + '_1_bn')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters, kernel_size, padding='same', name=name + '_2_conv')(x)
    x = BatchNormalization(name=name + '_2_bn')(x)

    x = Add()([shortcut, x])
    x = LeakyReLU(alpha=0.2)(x)
    return x

def build_model(input_shape=(224, 224, 3), num_classes=None):
    if num_classes is None:
        raise ValueError("num_classes must be provided")
    
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = residual_block(x, 64, stride=1, name='block1')
    x = residual_block(x, 128, stride=2, name='block2')
    x = residual_block(x, 256, stride=2, name='block3')
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)  # Add gradient clipping
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model