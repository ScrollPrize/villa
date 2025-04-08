from PIL import Image
# max image size None
Image.MAX_IMAGE_PIXELS = None
import os
import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description="Resize PNG images to 10% of their original size.")
parser.add_argument(
    "-d", "--directory", type=str, default=".", help="Directory containing PNG images to resize."
)
args = parser.parse_args()
# Check if the directory exists
if not os.path.exists(args.directory):
    print(f"Directory {args.directory} does not exist.")
    exit(1)
# Change to the specified directory
os.chdir(args.directory)

# Create a directory for the resized images if it doesn't exist
output_dir = "small"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all PNG files in the current directory
files = os.listdir(".")
sorted_files = sorted(files)
for file in sorted_files:
    if file.endswith(".png"):
        try:
            # Open the image
            with Image.open(file) as img:
                # Calculate the new size (10% of the original)
                new_size = (img.width // 10, img.height // 10)
                # Resize the image
                img_resized = img.resize(new_size, Image.ANTIALIAS)
                # Save the image in the output directory with the _small suffix
                img_resized.save(os.path.join(output_dir, file.replace(".png", "_small.png")))
                print(f"Resized and saved: {file}")
        except Exception as e:
            print(f"Failed to process {file}: {e}")
