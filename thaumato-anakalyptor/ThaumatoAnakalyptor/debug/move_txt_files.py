import os
import random
import shutil
import sys

def move_txt_files(directory, percentage):
    # Ensure the percentage is between 0 and 100
    if not 0 < percentage <= 100:
        print("Percentage must be between 0 and 100.")
        return

    # Get all txt files in the directory
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt') and os.path.isfile(os.path.join(directory, f))]
    
    # Calculate how many files to move based on the percentage
    num_files_to_move = int(len(txt_files) * (percentage / 100))
    
    # Randomly select files to move
    files_to_move = random.sample(txt_files, num_files_to_move)
    
    # Create the "_hibernating" directory if it doesn't exist
    hibernating_dir = os.path.join(directory, "_hibernating")
    if not os.path.exists(hibernating_dir):
        os.makedirs(hibernating_dir)

    # Move the selected files
    for file in files_to_move:
        shutil.move(os.path.join(directory, file), os.path.join(hibernating_dir, file))
        print(f"Moved: {file}")

    print(f"Successfully moved {len(files_to_move)} file(s) to {hibernating_dir}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python move_txt_files.py <directory> <percentage>")
    else:
        dir_path = sys.argv[1]
        try:
            percentage = float(sys.argv[2])
            move_txt_files(dir_path, percentage)
        except ValueError:
            print("Percentage must be a valid number.")