import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(image_dir, labels_file):
    # Read the labels file
    df = pd.read_csv(labels_file, sep=':', header=None, names=['filename', 'category'])
    df['filename'] = df['filename'].str.strip()
    df['category'] = df['category'].str.strip()
    
    # Create a dictionary mapping categories to integer labels
    categories = df['category'].unique()
    cat_to_label = {cat: i for i, cat in enumerate(categories)}
    
    # Add integer labels to the dataframe
    df['label'] = df['category'].map(cat_to_label)
    
    # Split the data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['category'], random_state=42)
    
    # Create ImageDataGenerator instances
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=image_dir,
        x_col='filename',
        y_col='category',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=image_dir,
        x_col='filename',
        y_col='category',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    return train_generator, val_generator, cat_to_label

if __name__ == "__main__":
    image_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    labels_file = os.path.join(os.path.dirname(__file__), "..", "classified_tool_names.txt")
    train_generator, val_generator, cat_to_label = load_data(image_dir, labels_file)
    print(f"Number of training samples: {len(train_generator.filenames)}")
    print(f"Number of validation samples: {len(val_generator.filenames)}")
    print(f"Number of categories: {len(cat_to_label)}")