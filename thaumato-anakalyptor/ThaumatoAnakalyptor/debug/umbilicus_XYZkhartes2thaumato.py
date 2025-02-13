# Load the input file
input_file = "khartes_umbilicus.txt"
output_file = "umbilicus_thaumato.txt"

# Read the content of the input file
with open(input_file, "r") as file:
    lines = file.readlines()

# Process the data
modified_data = []
axis_swap = [1, 2, 0]  # Define the axis swap order

for line in lines:
    # Split the line into integers
    values = list(map(float, line.strip().split(",")))
    
    # Swap the axes according to axis_swap
    swapped_values = [int(values[i]) for i in axis_swap]
    
    # Add 500 to each value
    modified_values = [value + 500 for value in swapped_values]
    
    # Format back to comma-separated string
    modified_data.append(", ".join(map(str, modified_values)))

# Save to the output file
with open(output_file, "w") as file:
    file.write("\n".join(modified_data))

print(f"Data successfully swapped, modified, and saved to {output_file}.")
