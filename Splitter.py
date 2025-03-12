import os
import random
import shutil

def shuffle_and_move_files(source_dir, train_dir, val_dir, num_files_to_move):
    # Ensure the target directory exists
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # List all files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Shuffle the list of files
    random.shuffle(files)

    # Select a subset of files
    selected_files = files[:num_files_to_move]

    # Move the selected files to the target directory
    for file_name in selected_files:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(val_dir, file_name)
        shutil.copy(source_path, target_path)

    selected_files = files[num_files_to_move:]

    # Move the selected files to the target directory
    for file_name in selected_files:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(train_dir, file_name)
        shutil.copy(source_path, target_path)


# Example usage
source_directory = "data\Fumo\Cirno"
train_dir = "data\Fumo\Train\Cirno"
val_dir = "data\Fumo\Val\Cirno"
number_of_files_to_move = 10  # Change this to the number of files you want to move

shuffle_and_move_files(source_directory, train_dir, val_dir, number_of_files_to_move)

# Example usage
source_directory = "data\Fumo\Miku"
train_dir = "data\Fumo\Train\Miku"
val_dir = "data\Fumo\Val\Miku"
number_of_files_to_move = 10  # Change this to the number of files you want to move

shuffle_and_move_files(source_directory, train_dir, val_dir, number_of_files_to_move)

# Example usage
source_directory = "data\Fumo\Rei"
train_dir = "data\Fumo\Train\Rei"
val_dir = "data\Fumo\Val\Rei"
number_of_files_to_move = 10  # Change this to the number of files you want to move

shuffle_and_move_files(source_directory, train_dir, val_dir, number_of_files_to_move)