import os

# Specify the folder path
folder_path = "/path/to/your/folder"

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    # Get the file name and extension
    name, ext = os.path.splitext(filename)
    
    # Create the new file name
    new_filename = f"{name}_0000{ext}"
    
    # Construct full file paths
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_filename)
    
    # Rename the file
    os.rename(old_file, new_file)
    print(f"Renamed: {filename} -> {new_filename}")

print("All files have been renamed.")