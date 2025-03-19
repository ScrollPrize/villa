import argparse

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Swap axes, add an offset to each element, and save the modified data."
    )
    parser.add_argument(
        "-i", "--input", type=str, default="khartes_umbilicus.txt",
        help="Path to the input file (default: khartes_umbilicus.txt)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="umbilicus_thaumato.txt",
        help="Path to the output file (default: umbilicus_thaumato.txt)"
    )
    parser.add_argument(
        "--swap", type=int, nargs=3, default=[1, 2, 0],
        help="Axis swap order as three integers (default: 1 2 0)"
    )
    parser.add_argument(
        "--offset", type=int, default=500,
        help="Offset value to add to each element (default: 500)"
    )

    args = parser.parse_args()

    # Load the input file
    with open(args.input, "r") as file:
        lines = file.readlines()

    modified_data = []

    # Process each line
    for line in lines:
        # Split the line into floats
        values = list(map(float, line.strip().split(",")))
        
        # Swap the axes according to the provided order and cast to int
        swapped_values = [int(values[i]) for i in args.swap]
        
        # Add the offset to each value
        modified_values = [value + args.offset for value in swapped_values]
        
        # Format back to a comma-separated string
        modified_data.append(", ".join(map(str, modified_values)))

    # Save the modified data to the output file
    with open(args.output, "w") as file:
        file.write("\n".join(modified_data))

    print(f"Data successfully swapped, modified, and saved to {args.output}.")

if __name__ == "__main__":
    main()
