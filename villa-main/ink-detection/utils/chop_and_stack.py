# This script chops an input image into multiple pieces, stacks them, and extends the context for each row.
# The simple stack is created by cropping the image into equal vertical strips and stacking them vertically.
# The extended stack adds a certain percentage of context from the next/previous rows, marked with 5% opacity yellow.

# Guide to use this script:
# 1. Ensure you have the Pillow library installed: `pip install Pillow`.
# 2. Save this script in a Python file, e.g., chop_and_stack.py.
# 3. Run the script from the command line with the following arguments:
#    - input_file: Path to the input image file.
#    - simple_output_file: Path to save the simple stacked output image.
#    - extended_output_file: Path to save the extended stacked output image.
#    - num_pieces: Number of vertical pieces to split the image into.
#    - --percent (optional): Percentage for right and left sections (default: 20).
# Example usage:
#    `python3 chop_and_stack.py input.jpg simple_output.jpg extended_output.jpg 4 --percent 20`

from PIL import Image, ImageDraw, ImageEnhance
import sys
import argparse

def chop_and_stack(input_file, simple_output_file, extended_output_file, num_pieces, percent=20):
    # Open the image file
    img = Image.open(input_file)
    width, height = img.size
    
    # Calculate the width of each piece
    piece_width = width // num_pieces

    # Create a new image to stack the pieces (simple-stack)
    simple_stack = Image.new('RGB', (piece_width, height * num_pieces))

    for i in range(num_pieces):
        # Calculate the box to crop
        box = (i * piece_width, 0, (i + 1) * piece_width, height)
        piece = img.crop(box)

        # Paste the piece into the new image
        simple_stack.paste(piece, (0, i * height))

    # Save the simple-stack image
    simple_stack.save(simple_output_file)

    # Create a new image for the extended-stack
    extended_width = int(piece_width * (1 + 2 * percent / 100))
    extended_stack = Image.new('RGB', (extended_width, height * num_pieces))

    # Copy and paste the specified sections
    right_percent_width = int(piece_width * percent / 100)
    left_percent_width = int(piece_width * percent / 100)
    middle_section_width = piece_width

    for i in range(1, num_pieces):
        # Copy the right percentage of the previous row
        right_section = simple_stack.crop((piece_width - right_percent_width, (i - 1) * height, piece_width, i * height))
        # Add yellow fill with 5% opacity
        yellow_overlay = Image.new('RGBA', right_section.size, (255, 255, 0, 10))
        right_section = Image.alpha_composite(right_section.convert('RGBA'), yellow_overlay)
        extended_stack.paste(right_section.convert('RGB'), (0, i * height))

    for i in range(num_pieces - 1):
        # Copy the left percentage of the next row
        left_section = simple_stack.crop((0, (i + 1) * height, left_percent_width, (i + 2) * height))
        # Add yellow fill with 5% opacity
        yellow_overlay = Image.new('RGBA', left_section.size, (255, 255, 0, 10))
        left_section = Image.alpha_composite(left_section.convert('RGBA'), yellow_overlay)
        extended_stack.paste(left_section.convert('RGB'), (extended_width - left_percent_width, i * height))

    # Paste the simple stack in the middle of the extended stack
    for i in range(num_pieces):
        piece = simple_stack.crop((0, i * height, piece_width, (i + 1) * height))
        extended_stack.paste(piece, (right_percent_width, i * height))

    # Fill the gap on the left of the first row with black
    black_section = Image.new('RGB', (right_percent_width, height), (0, 0, 0))
    extended_stack.paste(black_section, (0, 0))

    # Fill the gap on the right of the last row with black
    black_section = Image.new('RGB', (left_percent_width, height), (0, 0, 0))
    extended_stack.paste(black_section, (extended_width - left_percent_width, (num_pieces - 1) * height))

    # Add dashed lines to mark the edges of the simple stack
    draw = ImageDraw.Draw(extended_stack)
    dash_length = 40
    gap_length = 30
    for i in range((height * num_pieces) // (dash_length + gap_length)):
        start_y = i * (dash_length + gap_length)
        end_y = start_y + dash_length
        draw.line([(right_percent_width, start_y), (right_percent_width, end_y)], fill=(255, 255, 0), width=2)
        draw.line([(right_percent_width + piece_width, start_y), (right_percent_width + piece_width, end_y)], fill=(255, 255, 0), width=2)

    # Save the extended-stack image
    extended_stack.save(extended_output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chop and stack an image.')
    parser.add_argument('input_file', type=str, help='Input image file')
    parser.add_argument('simple_output_file', type=str, help='Simple stack output image file')
    parser.add_argument('extended_output_file', type=str, help='Extended stack output image file')
    parser.add_argument('num_pieces', type=int, help='Number of pieces to split the image into')
    parser.add_argument('--percent', type=int, default=20, help='Percentage for right and left sections (default: 20)')
    
    args = parser.parse_args()
    
    chop_and_stack(args.input_file, args.simple_output_file, args.extended_output_file, args.num_pieces, args.percent)
