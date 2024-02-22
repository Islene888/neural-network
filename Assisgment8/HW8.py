from sklearn.model_selection import train_test_split
import os
import shutil

# Assuming images are in a directory named 'dataset', organized into subdirectories for each class
dataset_dir = 'HW8 Dataset'

# Create directories for the split dataset if they don't exist
split_dirs = ['train', 'val', 'test']
for d in split_dirs:
    if not os.path.exists(d):
        os.makedirs(d)

# Split data into training, validation, and testing sets (70%, 15%, 15%)
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_dir)

    train_val_images, test_images = train_test_split(images, test_size=0.15, random_state=42)
    train_images, val_images = train_test_split(train_val_images, test_size=0.176,
                                                random_state=42)  # 0.176 is about 15% of 0.85


    # Function to copy images to their respective directories
    def copy_images(images, src_dir, dest_dir):
        for image in images:
            src_path = os.path.join(src_dir, image)
            dest_path = os.path.join(dest_dir, image)
            shutil.copy(src_path, dest_path)


    # Copy images to respective split directories
    for img_set, split_dir in zip([train_images, val_images, test_images], split_dirs):
        dest_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        copy_images(img_set, class_dir, dest_dir)

print("Dataset successfully split into training, validation, and testing sets.")
