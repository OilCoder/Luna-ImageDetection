import os
import re

def safe_rename(old_name, new_name, directory):
    """Rename file, appending a number if the new name already exists."""
    base, extension = os.path.splitext(new_name)
    counter = 1
    while os.path.exists(os.path.join(directory, new_name)):
        new_name = f"{base}_{counter}{extension}"
        counter += 1
    return new_name

def rename_files(directory):
    # Regex pattern to match the prefix we want to remove
    pattern = re.compile(r'^py\d+[-_]')
    
    # List to store new file names
    new_names = []

    for filename in os.listdir(directory):
        if pattern.match(filename):
            # Remove the prefix
            new_name = pattern.sub('', filename)
            
            # Ensure the new name is unique
            new_name = safe_rename(filename, new_name, directory)
            
            # Full paths for old and new names
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_name)
            
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} -> {new_name}")
            
            # Add new name to the list
            new_names.append(new_name)
        else:
            # If the file doesn't match the pattern, keep the original name
            new_names.append(filename)

    # Write all file names to a text file
    txt_file_path = os.path.join(os.path.dirname(directory), 'file_names.txt')
    with open(txt_file_path, 'w') as f:
        for name in new_names:
            f.write(f"{name}\n")
    
    print(f"All file names have been written to {txt_file_path}")

if __name__ == "__main__":
    # Specify the directory where your images are stored
    image_directory = "images"
    rename_files(image_directory)