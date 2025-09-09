
# Crackle-Viewer

<img src="crackle_viewer.png" alt="Crackle-Viewer Logo" width="200"/>

## Awards
This project won multiple awards in the [Vesuvius Challenge 2023](https://scrollprize.org/):

- 3rd place entry for the [Ink Detection Followup Prize for the Vesuvius Challenge 2023](https://scrollprize.substack.com/p/ink-detection-followup-prize-winners).

- Winning entry for the [Vesuvius Challenge Open Source Prizes December 2023](https://scrollprize.substack.com/p/open-source-prizes-awarded).

## Overview

This tool is designed to assist researchers and enthusiasts in inspecting and labeling ink residue on virtually unwrapped scrolls that were carbonized during the Vesuvius eruption nearly 2000 years ago. The primary goal is to generate a ground-truth ink dataset to make these virtually unwrapped Vesuvius scrolls readable, as it's difficult to discern ink residue by the naked eye.

This project was developed as part of the **Vesuvius Challenge 2023** and aims to contribute towards efforts in deciphering ancient texts carbonized during the Vesuvius eruption nearly 2000 years ago.

## Acknowledgment

This tool is based on the [PythonImageViewer](https://github.com/ImagingSolution/PythonImageViewer) and is distributed under the Apache 2.0 License.

## Features

### User Interface

- **Menu Bar**: Provides options to open images and access help.
- **Overlay Controls**: A comprehensive set of buttons, sliders, and controls for overlay manipulation.
- **Canvas**: A canvas area to display and interact with images.
- **Status Bar**: Displays image information and pixel coordinates.

### Functionalities

- **Image Navigation**: Open, zoom, rotate, translate and navigate through images.
- **Overlay Manipulation**: Load existing overlays or create new ones.
- **Labeling**: Draw on overlays to label ink residues.
- **Overlay Management**: Save individual or combined overlays.
- **Sub-Overlays**: Load and manage additional read-only overlays.
- **Opacity Control**: Adjust the opacity for overlays and sub-overlays.
- **Color Control**: Toggle or pick custom drawing colors.
- **Pencil Size**: Adjust the pencil size for drawing.

Please read trough the "Help" menu for more information.

### Keyboard and Mouse Shortcuts

- Use the Ctrl key while dragging to draw on the overlay, additionally you can also use the right mouse to draw.
- Double-click to fit the image to the screen and reset to intial orientation and image slice.
- Mouse wheel for zooming, Ctrl + Mouse wheel to rotate the image.
- Spacebar to toggle overlay visibility.
- "R" key to reset to the middle image.
- "C" key to toggle the drawing color.

## Requirements

- Python 3
- Numpy
- PIL (Pillow)
- Tkinter version 8.6

## Installation

1. First, install tkinter:
    ```bash
    sudo apt-get install python3-tk
    ```

2. Clone the repository and navigate to its directory:
    ```bash
    git clone https://github.com/schillij95/Crackle-Viewer
    cd Crackle-Viewer
    ```

3. Install the required Python packages:
    ```bash
    pip3 install -r requirements.txt
    ```

## Usage

### Download Executable

1. Go to the [Latest Release](https://github.com/schillij95/Crackle-Viewer/releases/tag/latest-release) and download the .zip for your operating system.
2. Unpack the .zip folder.
3. Start the executable inside the unpacked folder.
   
### Basic Usage

1. To run the tool using Python:
    ```bash
    python3 view_gui.py
    ```

### Advanced Usage (Executable)

1. First, make the `crackle_viewer` executable:
    ```bash
    chmod +x crackle_viewer
    ```
    
2. To run the tool, execute:
    ```bash
    ./crackle_viewer
    ```

## Building Executable (Optional)

1. Delete `build` and `dist` folders if they exist.
2. Install pyinstaller
    ```bash
    pip3 install pyinstaller==5.13.0
    ```
3. Run the following command to build an executable:
    ```bash
    pyinstaller --onefile --name crackle_viewer --hidden-import=PIL._tkinter_finder --add-data "crackle_viewer.png:." view_gui.py
    ```
    or
    ```bash
    pyinstaller crackle_viewer.spec
    ```

## Help

For details on the functionalities and controls, please refer to the `Help` menu within the application.

The Crackle Viewer was tested on Ubuntu 20.04.

## Contributing

Feel free to contribute to this project. Fork the repository, make your changes, and create a pull request.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

