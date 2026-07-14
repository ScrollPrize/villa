import argparse
import numpy as np
import struct

def save_npz_to_binary(npz_file, output_file):
    # Load the .npz file
    data = np.load(npz_file)
    
    # Extract f_star and deleted arrays from the npz file
    f_star_array = data['nodes_f']
    deleted_array = data['nodes_d']
    
    # Ensure compatibility with expected data types
    f_star_array = f_star_array.astype(np.float32)
    deleted_array = deleted_array.astype(bool)

    try:
        with open(output_file, 'wb') as outfile:
            # Write the number of nodes
            num_nodes = f_star_array.size
            outfile.write(struct.pack('I', num_nodes))  # 'I' is for unsigned int
            
            # Write each node's f_star and deleted status
            for f_star, deleted in zip(f_star_array, deleted_array):
                outfile.write(struct.pack('f', f_star))   # 'f' is for float
                outfile.write(struct.pack('?', deleted))  # '?' is for bool

    except IOError as e:
        print(f"Error opening file for writing: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert .npz file to binary format.")
    parser.add_argument("npz_file", type=str, help="Input .npz file containing f_star and deleted arrays.")
    parser.add_argument("output_file", type=str, help="Output binary file.")
    
    args = parser.parse_args()
    
    # Call the save function with provided arguments
    save_npz_to_binary(args.npz_file, args.output_file)

if __name__ == "__main__":
    main()
