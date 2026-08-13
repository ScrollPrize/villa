# Probably out-of-data. But not sure.

import os

# Function to update the 8th entry of each line
def update_line(line, line_number):
    if line_number < 5:  # Only update the first 5 lines
        entries = line.split(",")  # Split the line by comma
        if len(entries) >= 7:  # Make sure there are enough entries
            entries[6] = entries[6].replace('1.000000000000000000e+00', '2.000000000000000000e+00')  # Update 8th entry
        return ",".join(entries) + "\n"  # Rejoin the entries and return
    else:
        return line + "\n"  # No update required, return the line as is

# List all .txt files in the current directory
for filename in os.listdir("train"):
    if filename.endswith(".txt"):
        with open("train/" + filename, 'r') as f:
            lines = f.readlines()  # Read all lines from file

        # Remove new line characters from each line
        lines = [line.strip() for line in lines]

        # Update lines
        updated_lines = [update_line(lines[i], i) for i in range(len(lines))]

        # Write updated lines back to file
        with open("train/" + filename, 'w') as f:
            f.writelines(updated_lines)

        print(f"Updated {filename}")
