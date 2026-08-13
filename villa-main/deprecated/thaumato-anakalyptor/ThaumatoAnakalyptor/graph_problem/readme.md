# How to Use:
### Create a Directory Structure:

css
Copy code
graph_problem/
├── CMakeLists.txt
└── main.cpp
└── example.cpp
Build the Project:

bash
Copy code
mkdir build
cd build
cmake ..
make
This will generate the graph_problem executable in the build directory.

## Run the Executable

bash:
python3 graph_solve.py ARGS

The example.cpp holds a clean template.
main.cpp contains the current solution based loosely on Believe Propagation.