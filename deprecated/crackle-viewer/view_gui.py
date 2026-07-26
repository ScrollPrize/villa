
### Julian Schilliger - Crackle Viewer - Vesuvius Challenge 2023

import tkinter as tk
import tkinter.colorchooser
import threading
from collections import deque
from tkinter import filedialog
import textwrap
from PIL import Image, ImageTk, ImageDraw, ImageChops, ImageEnhance
# Increase the image pixel limit to the desired value
# Image.MAX_IMAGE_PIXELS = 300000000
Image.MAX_IMAGE_PIXELS = None
import math
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
from multiprocessing import Pool
from scipy.spatial import KDTree
import open3d as o3d
import cv2

def load_image_disk(filename):
    pil_image = np.array(Image.open(filename))
    # Convert to 8-bit and grayscale if needed
    if pil_image.dtype == np.uint16:
        pil_image = np.uint16(pil_image//256)
    return pil_image

def load_image_parallel(filename):
    return filename, load_image_disk(filename)

def compute_uv_bounding_box(uv_vertices):
    """Compute the bounding box of a triangle in UV space."""
    min_uv = np.min(uv_vertices, axis=0)
    max_uv = np.max(uv_vertices, axis=0)
    return min_uv, max_uv

def preprocess_uv_triangles(mesh):
    """Preprocess the mesh to create a KDTree with UV bounding boxes."""
    bounding_boxes = []
    triangle_data = []
    min_uv_ = np.array([1000, 1000])
    max_uv_ = np.array([-10000, -10000])
    triangle_uvs = np.array(mesh.triangle_uvs).reshape((-1,3,2))
    
    for i, triangle in enumerate(mesh.triangles):
        # Get the UV coordinates of the vertices of this triangle
        uv_vertices = triangle_uvs[i]
        
        # Calculate the bounding box for this triangle in UV space
        min_uv, max_uv = compute_uv_bounding_box(uv_vertices)
        min_uv_ = np.minimum(min_uv_, min_uv)
        max_uv_ = np.maximum(max_uv_, max_uv)
        
        # Use the center of the bounding box as the KDTree point and store the data
        center_uv = (min_uv + max_uv) / 2
        bounding_boxes.append(center_uv)
        triangle_data.append((triangle, uv_vertices, min_uv, max_uv))
    
    print(f"UV bounding box: {min_uv_} - {max_uv_}")
    # Create KDTree for the bounding boxes' centers
    kd_tree = KDTree(bounding_boxes)
    return kd_tree, triangle_data

def barycentric_coordinates(p, a, b, c):
    # Calculate vectors from a to b, a to c, and a to p
    v0 = b - a
    v1 = c - a
    v2 = p - a

    # Compute dot products
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    # Compute denominator
    denom = d00 * d11 - d01 * d01

    # Barycentric coordinates
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

def find_uv_triangle(mesh_vertices, uv_point, kd_tree, triangle_data):
    """Find the 3D point corresponding to a 2D UV point by querying preprocessed triangles."""
    # Query KDTree for nearby bounding boxes
    _, idxs = kd_tree.query(uv_point, k=500)  # Adjust `k` as needed for balance between precision and performance
    
    for idx in idxs:
        triangle, uv_vertices, min_uv, max_uv = triangle_data[idx]
        
        # First, check if the point is within the bounding box in UV space
        if np.all(uv_point >= min_uv) and np.all(uv_point <= max_uv):
            # Calculate barycentric coordinates
            u, v, w = barycentric_coordinates(uv_point, uv_vertices[0], uv_vertices[1], uv_vertices[2])
            if u >= 0 and v >= 0 and w >= 0:
                # Retrieve the 3D vertices and interpolate
                vertices = np.array([mesh_vertices[i] for i in triangle])
                point_3d = u * vertices[0] + v * vertices[1] + w * vertices[2]
                return point_3d
    
    # If no triangle contains the uv point
    return None

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.my_title = "Vesuvius Crackle Viewer"
        self.master.title(self.my_title)
        if getattr(sys, 'frozen', False):
            # Running as compiled
            base_path = sys._MEIPASS
        else:
            # Running as script
            base_path = os.path.dirname(__file__)
        icon_path = os.path.join(base_path, 'crackle_viewer.png')
        # Set the window icon
        icon = tk.PhotoImage(file=icon_path)   
        self.master.tk.call('wm', 'iconphoto', self.master._w, icon)
        self.master.geometry("800x600")
        self.master.minsize(width=800, height=600)
        self.load_last_directory()  # Load the last directory
        self.images_folder = ""
        self.pil_image = None
        self.min_value = 0.0
        self.max_value = 65535.0
        self.sub_overlays = []
        self.sub_overlay_colors = ['white', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        self.sub_overlay_names = ['overlay.png']
        self.current_sub_overlay = tk.StringVar() 
        self.current_sub_overlay.set("None") 
        self.current_sub_overlay.trace("w", self.suboverlay_selected)
        self.create_menu()
        self.create_widget()
        self.reset_transform()
        self.image_list = []
        self.image_index = 0
        self.last_directory_overlay = None
        self.last_directory_suboverlay = None
        self.cursor_circle = None
        self.shift_pressed = False
        self.mouse_is_pressed = False
        self.overlay_image = None
        self.overlay_visibility = tk.BooleanVar(value=True)
        self.pencil_color = 'white'
        self.pencil_size = 45
        self.flood_fill_active = False
        self.ff_threshold = 10
        self.max_propagation_steps = 10
        self.global_scale_factor = 1.0
        self.micron_factor = 0.00324
        self.resampling_methods = {
            "NEAREST": Image.Resampling.NEAREST,
            "BILINEAR": Image.Resampling.BILINEAR,
            "BICUBIC": Image.Resampling.BICUBIC,
        }
        self.resample_method = tk.StringVar(value="NEAREST")

        self.create_overlay_controls()

    def menu_open_clicked(self, event=None):
        self.load_images()
        try:
            self.load_obj()
        except:
            pass

    def menu_quit_clicked(self):
        self.master.destroy() 

    def create_menu(self):
        self.menu_bar = tk.Menu(self) 
 
        self.file_menu = tk.Menu(self.menu_bar, tearoff = tk.OFF)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.file_menu.add_command(label="Open", command = self.menu_open_clicked, accelerator="Ctrl+O")
        self.file_menu.add_separator() 
        self.file_menu.add_command(label="Exit", command = self.menu_quit_clicked)

        self.menu_bar.bind_all("<Control-o>", self.menu_open_clicked)

        self.help_menu = tk.Menu(self.menu_bar, tearoff=tk.OFF)
        self.menu_bar.add_command(label="Help", command=self.show_help)

        self.master.config(menu=self.menu_bar)

    def create_overlay_controls(self):
        self.overlay_frame = tk.Frame(self.master)
        self.overlay_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.overlay_btn = tk.Button(self.overlay_frame, text="Load Overlay", command=self.load_overlay_image)
        self.overlay_btn.pack(side=tk.LEFT)

        self.create_empty_image_btn = tk.Button(self.overlay_frame, text="Create Empty Image", command=self.create_empty_overlay_image)
        self.create_empty_image_btn.pack(side=tk.LEFT, padx=5)


        self.save_btn = tk.Button(self.overlay_frame, text="Save Overlay", command=self.save_overlay)
        self.save_btn.pack(side=tk.LEFT)

        self.save_combined_btn = tk.Button(self.overlay_frame, text="Save Combined Overlays", command=self.save_combined_overlays)
        self.save_combined_btn.pack(side=tk.LEFT)

        self.save_displayed_btn = tk.Button(self.overlay_frame, text="Save Displayed Image", command=self.save_displayed_image)
        self.save_displayed_btn.pack(side=tk.LEFT)

        
        self.overlay_check = tk.Checkbutton(self.overlay_frame, text="Show Overlay", variable=self.overlay_visibility, command=self.toggle_overlay)
        self.overlay_check.pack(side=tk.LEFT)
        
        self.color_btn = tk.Button(self.overlay_frame, text="Toggle Color", command=self.toggle_color)
        self.color_btn.pack(side=tk.LEFT)

        self.color_label = tk.Label(self.overlay_frame, text="", width=5, background=self.pencil_color)
        self.color_label.pack(side=tk.LEFT, padx=5)
        self.color_label.bind("<Button-1>", lambda e: self.toggle_color()) # Add this line in create_overlay_controls method

        # Add slider for overlay opacity control
        self.overlay_opacity_scale = tk.Scale(self.overlay_frame, from_=0, to_=255, orient=tk.HORIZONTAL, label="Overlay Opacity", length=175)
        self.overlay_opacity_scale.bind("<Motion>", self.adjust_overlay_opacity) # Bind the slider's motion to adjust opacity
        self.overlay_opacity_scale.pack(side=tk.LEFT)
        self.overlay_opacity_scale.config(from_=0, to=1, resolution=0.01)
        self.overlay_opacity_scale.set(1.0)

        # Entry to set opacity value manually
        self.overlay_opacity_entry = tk.Entry(self.overlay_frame, width=5)
        self.overlay_opacity_entry.pack(side=tk.LEFT, padx=5)
        self.overlay_opacity_entry.insert(tk.END, '1.0')
        self.overlay_opacity_entry.bind('<Return>', self.set_opacity_from_entry)

        self.pick_color_btn = tk.Button(self.overlay_frame, text="Pick Color", command=self.pick_color)
        self.pick_color_btn.pack(side=tk.LEFT)
        
        self.size_scale = tk.Scale(self.overlay_frame, from_=1, to_=500, orient=tk.HORIZONTAL, label="Pencil Size", length=125)
        self.size_scale.set(self.pencil_size)
        self.size_scale.pack(side=tk.LEFT)

        self.suboverlay_frame = tk.Frame(self.master)  # Create a new frame for SubOverlay-related controls
        self.suboverlay_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)  # Pack it below the existing frame

        # Dropdown to select a sub-overlay
        self.suboverlay_label = tk.Label(self.suboverlay_frame, text="Select Overlay:")
        self.suboverlay_label.pack(side=tk.LEFT)

        self.select_suboverlay_optionmenu = tk.OptionMenu(self.suboverlay_frame, self.current_sub_overlay, "None", *self.sub_overlay_names)
        self.select_suboverlay_optionmenu.pack(side=tk.LEFT)

        self.add_suboverlay_btn = tk.Button(self.suboverlay_frame, text="Add SubOverlay", command=self.load_suboverlay)
        self.add_suboverlay_btn.pack(side=tk.LEFT)

        self.clear_suboverlays_btn = tk.Button(self.suboverlay_frame, text="Clear SubOverlays", command=self.clear_suboverlays)
        self.clear_suboverlays_btn.pack(side=tk.LEFT)

        # make sure to add enough space to display full name of the suboverlay
        self.suboverlay_opacity_scale = tk.Scale(self.suboverlay_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="SubOverlay Opacity", length=175)
        self.suboverlay_opacity_scale.bind("<Motion>", self.adjust_suboverlay_opacity)
        self.suboverlay_opacity_scale.pack(side=tk.LEFT)
        self.suboverlay_opacity_scale.config(from_=0, to=1, resolution=0.01)
        self.suboverlay_opacity_scale.set(0.4)
        # spaceing between the two scales
        tk.Label(self.suboverlay_frame, text=" ").pack(side=tk.LEFT)

        self.suboverlay_brightness_scale = tk.Scale(self.suboverlay_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="SubOverlay Brightness", length=175)
        self.suboverlay_brightness_scale.bind("<Motion>", self.adjust_suboverlay_opacity)
        self.suboverlay_brightness_scale.pack(side=tk.LEFT)
        self.suboverlay_brightness_scale.config(from_=0, to=10.0, resolution=0.01)
        self.suboverlay_brightness_scale.set(1.0)

        # Dropdown menu for resampling method
        
        self.resample_method_label = tk.Label(self.suboverlay_frame, text="Resampling Method:")
        self.resample_method_label.pack(side=tk.LEFT)
        self.resample_method_optionmenu = tk.OptionMenu(
            self.suboverlay_frame, 
            self.resample_method, 
            *self.resampling_methods.keys(),
            command=self.on_resample_method_changed
        )
        self.resample_method_optionmenu.pack(side=tk.LEFT)

        # Set min and max values for the image
        self.minmax_label = tk.Label(self.overlay_frame, text="Min Max image values:")
        self.minmax_label.pack(side=tk.LEFT)
        self.min_value_entry = tk.Entry(self.overlay_frame, width=5)
        self.min_value_entry.pack(side=tk.LEFT, padx=5)
        self.min_value_entry.insert(tk.END, '0')
        self.min_value_entry.bind('<Return>', self.set_max_min_from_entry)

        self.max_value_entry = tk.Entry(self.overlay_frame, width=5)
        self.max_value_entry.pack(side=tk.LEFT, padx=5)
        self.max_value_entry.insert(tk.END, '65535')
        self.max_value_entry.bind('<Return>', self.set_max_min_from_entry)

        # micron factor for the image
        self.micron_label = tk.Label(self.overlay_frame, text="Micron Factor:")
        self.micron_label.pack(side=tk.LEFT)
        self.micron_entry = tk.Entry(self.overlay_frame, width=5)
        self.micron_entry.pack(side=tk.LEFT, padx=5)
        self.micron_entry.insert(tk.END, '0.00324')
        self.micron_entry.bind('<Return>', self.set_micron_factor)

        self.reset_slice_btn = tk.Button(self.overlay_frame, text="Reset Slice", command=self.reset_to_middle_image)
        self.reset_slice_btn.pack(side=tk.LEFT, padx=5)

        self.max_propagation_var = tk.IntVar(value=self.max_propagation_steps)
        max_propagation_label = tk.Label(self.suboverlay_frame, text="Max Propagation:")
        max_propagation_label.pack(side=tk.LEFT, padx=(10, 2))

        max_propagation_slider = tk.Scale(self.suboverlay_frame, from_=1, to=500, orient=tk.HORIZONTAL, command=self.update_max_propagation)
        max_propagation_slider.set(self.max_propagation_steps)
        max_propagation_slider.pack(side=tk.LEFT, padx=2)

        max_propagation_value_label = tk.Label(self.suboverlay_frame, textvariable=self.max_propagation_var)
        max_propagation_value_label.pack(side=tk.LEFT, padx=(0, 10))

        self.bucket_threshold_var = tk.StringVar(value="10")
        bucket_threshold_label = tk.Label(self.suboverlay_frame, text="FF Threshold:")
        bucket_threshold_label.pack(side=tk.LEFT, padx=(10, 2))

        self.bucket_threshold_slider = tk.Scale(self.suboverlay_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_threshold_value)
        self.bucket_threshold_slider.pack(side=tk.LEFT, padx=2)

        bucket_threshold_value_label = tk.Label(self.suboverlay_frame, textvariable=self.bucket_threshold_var)
        bucket_threshold_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # New Frame for Image Processing Controls
        self.image_processing_frame = tk.Frame(self.master)
        self.image_processing_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Operation Dropdown
        self.operation_var = tk.StringVar(value="max")
        self.operation_menu = tk.OptionMenu(
            self.image_processing_frame, 
            self.operation_var, 
            "max", "min", "mean",
            command=self.operation_changed
        )
        self.operation_menu.pack(side=tk.LEFT)

        # Radius Input
        self.radius_var = tk.StringVar(value="0")
        self.radius_entry = tk.Entry(
            self.image_processing_frame, 
            textvariable=self.radius_var,
            width=5
        )
        self.radius_entry.pack(side=tk.LEFT)
        self.radius_entry.bind('<Return>', self.update_radius_and_refocus)

        # Direction Radio Buttons
        self.direction_var = tk.StringVar(value="omi")
        directions = [("omi", "omi"), ("front", "front"), ("back", "back")]
        for text, mode in directions:
            tk.Radiobutton(
                self.image_processing_frame, 
                text=text, 
                variable=self.direction_var, 
                value=mode
            ).pack(side=tk.LEFT)

        self.direction_var.trace("w", lambda name, index, mode: self.process_images())

        # Preload Images Checkbox
        self.toggle_contrast_var = tk.BooleanVar(value=False)
        self.toggle_contrast_check = tk.Checkbutton(
            self.image_processing_frame, 
            text="Contrast Enhance", 
            variable=self.toggle_contrast_var,
            command=self.toggle_contrast
        )
        self.toggle_contrast_check.pack(side=tk.LEFT)

        # Preload Images Checkbox
        self.preload_images_var = tk.BooleanVar(value=False)
        self.preload_images_check = tk.Checkbutton(
            self.image_processing_frame, 
            text="Preload Images", 
            variable=self.preload_images_var,
            command=self.toggle_preload
        )
        self.preload_images_check.pack(side=tk.LEFT)

        self.layer_control_frame = tk.Frame(self.master)
        self.layer_control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.layer_index_label = tk.Label(self.layer_control_frame, text="Layer Index:")
        self.layer_index_label.pack(side=tk.LEFT, padx=5)

        self.layer_index_var = tk.StringVar(value=str(self.image_index))
        self.layer_index_entry = tk.Entry(self.layer_control_frame, textvariable=self.layer_index_var, width=6)
        self.layer_index_entry.pack(side=tk.LEFT, padx=5)
        self.layer_index_entry.bind('<Return>', self.set_layer_from_entry)

        self.set_layer_btn = tk.Button(self.layer_control_frame, text="Set Layer", command=self.set_layer_from_entry)
        self.set_layer_btn.pack(side=tk.LEFT, padx=5)

    def create_widget(self):

        frame_statusbar = tk.Frame(self.master, bd=1, relief = tk.SUNKEN)
        self.label_image_info = tk.Label(frame_statusbar, text="image info", anchor=tk.E, padx = 5)
        self.label_image_pixel = tk.Label(frame_statusbar, text="2D: (x, y) 3D: (x, y, z)", anchor=tk.W, padx = 5)
        self.label_image_info.pack(side=tk.RIGHT)
        self.label_image_pixel.pack(side=tk.LEFT)
        frame_statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas
        self.canvas = tk.Canvas(self.master, background="black")
        self.canvas.pack(expand=True,  fill=tk.BOTH)  

        self.canvas.bind("<Button-1>", self.mouse_down_left)                   # MouseDown
        self.master.bind("<ButtonRelease-1>", self.mouse_up_left)              # MouseUp
        self.canvas.bind("<B1-Motion>", self.mouse_move_left)                  # MouseDrag
        self.canvas.bind("<Button-3>", self.mouse_down_right)                  # MouseDown
        self.master.bind("<ButtonRelease-3>", self.mouse_up_right)             # MouseUp
        self.canvas.bind("<B3-Motion>", self.mouse_move_right)                 # MouseDrag
        self.master.bind("f", self.threaded_flood_fill)                        # FloodFill
        self.canvas.bind("<Motion>", self.mouse_move)                          # MouseMove
        self.canvas.bind("<Leave>", self.mouse_leave_canvas)                   # MouseLeave
        self.canvas.bind("<Double-Button-1>", self.mouse_double_click_left)    # MouseDoubleClick
        self.master.bind("<Control_L>", self.shift_press)
        self.master.bind("<KeyRelease-Control_L>", self.shift_release)
        self.master.bind("<Control_R>", self.shift_press)
        self.master.bind("<KeyRelease-Control_R>", self.shift_release)
        self.master.bind("<space>", lambda _: {self.overlay_visibility.set(not self.overlay_visibility.get()), self.redraw_image()})           # Spacebar
        self.master.bind("r", self.reset_to_middle_image)
        self.master.bind("c", self.toggle_color)

        
        if sys.platform == 'linux':  # Linux OS
            self.canvas.bind("<Button-4>", self.mouse_wheel)
            self.canvas.bind("<Button-5>", self.mouse_wheel)
        else:  # Windows OS
            self.master.bind("<MouseWheel>", self.mouse_wheel)

        # Bind left and right arrow keys for previous and next functionality
        self.master.bind("<Left>", self.show_previous_image)
        self.master.bind("<Left>", lambda event: self.show_previous_image(event, 5) if event.state & 0x1 else self.show_previous_image(event))
        self.master.bind("<Right>", self.show_next_image)
        self.master.bind("<Right>", lambda event: self.show_next_image(event, 5) if event.state & 0x1 else self.show_next_image(event))

    # Update radius and refocus method
    def update_radius_and_refocus(self, event=None):
        self.master.focus()  # Set focus back to the master window
        self.process_images()
 
    def show_help(self):
        help_message = textwrap.dedent("""
        Vesuvius Crackle Viewer Usage:

        - Open: Ctrl+O to open image.
        - Exit: Close the application.
        - Use the Ctrl key while dragging to draw on the overlay.
        - Double click to zoom fit.
        - Use the mouse wheel to zoom in/out.
        - Ctrl + Mouse wheel to rotate the image.
        - Space to toggle overlay visibility.
        - R to reset to the middle image.
        - C to toggle the drawing color.
        - Double click inside the image to reset the zoom, rotation and slice.
        - F to flood fill from the selected point

        Overlay Controls:
        - Load Overlay: Load an overlay image.
        - Create Empty Image: Create an empty overlay.
        - Save Overlay: Save the current overlay.
        - Save Combined Overlays: Save the combined image of the overlay and all sub-overlays.
        - Toggle Color: Switch between drawing colors.
        - Overlay Opacity: Adjust the opacity of the overlay.
        - Pick Color: Choose a custom drawing color.
        - Adjust the Pencil Size for drawing on the overlay.
        - Adjust the image brightness and contrast with the Min Max image values.
        - Adjust the micron factor for the image for the scale bar.
        - Dropdown to select active overlay for drawing.
        - Add SubOverlay: Load one or multiple SubOverlay image. These images are displayed only and not mutable.
        - Clear SubOverlays: Clear all SubOverlays.
        - SubOverlay Opacity: Adjust the opacity of the SubOverlay.
        - Max Propagation: Select the max numbers of points to color with flood fill
        - FF Threshold: Specify the threshold to color adjacent points with flood fill
        - Reset Slice: Reset to the middle image.
        - Composite image: Compose multiple tif images into one image. Can use min, max or mean operation. Can specify the number of slices and direction of the images to be composed.
        - Preload Images: Preload all images in the folder. This will speed up the navigation between images and composition of images.
        - Layer Index: Set the current image to the specified layer index.
        """)
        tk.messagebox.showinfo("Help", help_message)

    def save_last_directory(self):
        file_path = os.path.join(self.base_dir, "last_directory.txt")
        with open(file_path, "w") as file:
            file.write(self.last_directory)

    def load_last_directory(self):
        file_path = os.path.join(self.base_dir, "last_directory.txt")
        try:
            with open(file_path, "r") as file:
                self.last_directory = file.read().strip()
        except FileNotFoundError:
            self.last_directory = None

    def toggle_contrast(self):
        self.process_images()

    def toggle_preload(self):
        if self.preload_images_var.get():
            self.preload_all_images()
        else:
            self.flush_preloaded_images()

    def preload_all_images(self):
        self.preloaded_images = {}

        with Pool() as pool:
            results = list(tqdm(pool.imap(load_image_parallel, self.image_list), total=len(self.image_list)))
            
        for filename, image in results:
            self.preloaded_images[filename] = image

    ## Single threaded version
    # def preload_all_images(self):
    #     self.preloaded_images = {}
    #     for i, img_path in enumerate(tqdm(self.image_list)):
    #         self.preloaded_images[img_path] = self.load_image(img_path, as_np=True)

    def flush_preloaded_images(self):
        self.preloaded_images = {}

    def load_image(self, filename, as_np=False):
        if self.preload_images_var.get() and filename in self.preloaded_images:
            pil_image = self.preloaded_images[filename]
        else:
            pil_image = load_image_disk(filename)
        # pil_image = np.clip(pil_image, 0, 255)
        if not as_np:
            pil_image = Image.fromarray(np.uint8(pil_image)).convert("L")
        return pil_image
    
    def load_images(self):
        initial_dir =self.last_directory if self.last_directory else os.getcwd()
        images_path = tk.filedialog.askdirectory(
            initialdir = initial_dir
        )
        if images_path:
            self.last_directory = images_path
            # Create surface volume dir
            surface_volume_path = os.path.join(images_path, "layers")
            if not os.path.exists(surface_volume_path):
                 surface_volume_path = os.path.join(images_path, "surface_volume")
            print(surface_volume_path)
            self.images_folder = surface_volume_path.rsplit('/', 1)[-1]
            self.save_last_directory()  # Save the last_directory
            self.image_list = sorted(glob.glob(os.path.join(surface_volume_path, f'*.tif')))
            #hacky way to get png and jpg file stacks
            if len(self.image_list) == 0:
                self.image_list = sorted(glob.glob(os.path.join(surface_volume_path, f'*.png')))
            if len(self.image_list) == 0:
                self.image_list = sorted(glob.glob(os.path.join(surface_volume_path, f'*.jpg')))
            if len(self.image_list) == 0:
                print("No tif, png or jpg images found in the directory.")
            self.image_index = len(self.image_list) // 2
            
            if self.preload_images_var.get():
                self.preload_all_images()
            self.set_image(self.image_list[self.image_index])
    
    def load_obj(self):
        # Load the obj file(s) from the self.last_directory
        obj_files = sorted(glob.glob(os.path.join(self.last_directory, f'*.obj')))
        # filter the obj files with uv coordinates. load and check
        obj_file = None
        for obj in obj_files:
            # open3d open
            mesh = o3d.io.read_triangle_mesh(obj)
            if len(mesh.triangle_uvs) > 0:
                obj_file = obj
                break
        if obj_file:
            print(f"Loaded obj file: {obj_file}")
            self.mesh = o3d.io.read_triangle_mesh(obj_file)
            self.mesh_vertices = np.array(self.mesh.vertices)
            print("Preprocessing obj transformation ... ")
            self.kd_tree, self.triangle_data = preprocess_uv_triangles(self.mesh)
            print("Preprocessing obj transformation done.")

    def load_overlay_image(self):
        initial_dir = self.last_directory_overlay if self.last_directory_overlay else self.last_directory if self.last_directory else os.getcwd()
        try:
            file_path = tk.filedialog.askopenfilename(filetypes=[('PNG files', '*.png')], initialdir=initial_dir)
        except:
            file_path = tk.filedialog.askopenfilename(filetypes=[('PNG files', '*.png')], initialdir=os.getcwd())
        if file_path:
            self.last_directory_overlay = file_path
            print(file_path, self.last_directory_overlay)
            self.overlay_image = Image.open(file_path).convert("L")
            # self.overlay_image = Image.fromarray(np.uint8(np.array(Image.open(file_path)))).convert("L")
            if len(self.sub_overlays) == 0:
                self.sub_overlays.append(self.overlay_image)
            else:
                self.sub_overlays[0] = self.overlay_image
            self.redraw_image()
            # strip the file name from the path and save directory
            self.sub_overlay_names[0] = file_path.rsplit('/', 1)[-1]
            self.update_suboverlay_dropdown()

    def create_empty_overlay_image(self):
        if not self.image_list:
            return
        reference_image = Image.open(self.image_list[0])
        width, height = reference_image.size
        # Changed from RGBA to 'L' for grayscale and set initial color to black
        self.overlay_image = Image.new("L", (width, height), "black")
        if len(self.sub_overlays) == 0:
            self.sub_overlays.append(self.overlay_image)
        else:
            self.sub_overlays[0] = self.overlay_image
        self.sub_overlay_names[0] = "newly_created_overlay.png"
        self.update_suboverlay_dropdown()
        self.redraw_image()

    # Method to load SubOverlay
    def load_suboverlay(self):
        initial_dir = self.last_directory_suboverlay if self.last_directory_suboverlay else self.last_directory if self.last_directory else os.getcwd()
        try:
            file_path = tk.filedialog.askopenfilename(filetypes=[('PNG files', '*.png'), ('TIF files', '*.tif')], initialdir=initial_dir)
        except:
            file_path = tk.filedialog.askopenfilename(filetypes=[('PNG files', '*.png'), ('TIF files', '*.tif')], initialdir=os.getcwd())
        if file_path:
            self.last_directory_suboverlay = file_path
            if ".png" in file_path:
                sub_overlay = Image.open(file_path).convert("L")
            elif ".tif" in file_path:
                sub_overlay = Image.fromarray(np.uint8(np.array(Image.open(file_path))//256)).convert("L")
            else:
                raise ValueError("File type not supported.")
            self.sub_overlays.append(sub_overlay)
            self.redraw_image()
            # strip the file name from the path and save directory
            self.sub_overlay_names.append(file_path.rsplit('/', 1)[-1])
            if len(self.sub_overlay_names) >= len(self.sub_overlay_colors):
                self.sub_overlay_colors.append(self.sub_overlay_colors[len(self.sub_overlay_colors)-1])
            self.update_suboverlay_dropdown()

    def update_suboverlay_dropdown(self):
        menu = self.select_suboverlay_optionmenu['menu']
        menu.delete(0, 'end')
        new_choices = self.sub_overlay_names
        for choice in new_choices:
            menu.add_command(label=choice, command=tk._setit(self.current_sub_overlay, choice))

    def suboverlay_selected(self, *args):
        selected_name = self.current_sub_overlay.get()
        try:
            selected_index = self.sub_overlay_names.index(selected_name)

            # Swap the 0-th element with the selected_index element
            name0, name1 = self.sub_overlay_names[selected_index], self.sub_overlay_names[0]
            self.sub_overlay_names[0], self.sub_overlay_names[selected_index] = name0, name1

            # Swap the 0-th element with the selected_index element for colors as well
            color0, color1 = self.sub_overlay_colors[selected_index], self.sub_overlay_colors[0]
            self.sub_overlay_colors[0], self.sub_overlay_colors[selected_index] = color0, color1

            # Swap the 0-th element with the selected_index element for sub_overlays as well
            sub_overlay0, sub_overlay1 = self.sub_overlays[selected_index], self.sub_overlays[0]
            self.sub_overlays[0], self.sub_overlays[selected_index] = sub_overlay0, sub_overlay1

            self.overlay_image = self.sub_overlays[0]
            
            # Update the dropdown
            self.update_suboverlay_dropdown()
            
            # Set the current value to the new 0-th element
            self.current_sub_overlay.set(self.sub_overlay_names[0])

            self.redraw_image()
            self.toggle_color()
            self.toggle_color()

        except ValueError:
            print("Selected value is not in the list.")


    def save_overlay(self):
        if self.overlay_image:
            initial_dir = self.last_directory_overlay if self.last_directory_overlay else os.getcwd()
            try:
                save_path = tk.filedialog.asksaveasfilename(filetypes=[('PNG files', '*.png')], initialdir=initial_dir)
            except:
                save_path = tk.filedialog.asksaveasfilename(filetypes=[('PNG files', '*.png')], initialdir=os.getcwd())
            
            if save_path:
                # Convert to grayscale and remove alpha channel
                bw_image = self.overlay_image.convert("1")
                bw_image.save(save_path)

    def save_combined_overlays(self):
        if self.pil_image:
            # Create a base image
            combined = Image.new("L", (self.pil_image.width, self.pil_image.height), color="black")

            # Add sub-overlays
            for sub_overlay in self.sub_overlays:
                combined = ImageChops.lighter(combined, sub_overlay.convert("L"))

            # Add the main overlay
            if self.overlay_image:
                combined = ImageChops.lighter(combined, self.overlay_image.convert("L"))

            # Save the combined image in grayscale and without an alpha channel
            initial_dir = self.last_directory_overlay if self.last_directory_overlay else os.getcwd()
            try:
                save_path = tk.filedialog.asksaveasfilename(filetypes=[('PNG files', '*.png')], initialdir=initial_dir)
            except:
                save_path = tk.filedialog.asksaveasfilename(filetypes=[('PNG files', '*.png')], initialdir=os.getcwd())

            if save_path:
                bw_combined = combined.convert("1")
                bw_combined.save(save_path)

    def save_displayed_image(self):
        if self.pil_image is None:
            tk.messagebox.showerror("Error", "No image to save.")
            return

        initial_dir = self.last_directory_overlay if self.last_directory_overlay else os.getcwd()
        # Asks the user for the location and name of the file to save
        try:
            file_path = tk.filedialog.asksaveasfilename(
                defaultextension=".tif",
                filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")],
                initial_dir=initial_dir
            )
        except:
            file_path = tk.filedialog.asksaveasfilename(
                defaultextension=".tif",
                filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")],
                initialdir=os.getcwd()
            )

        if not file_path:
            # User cancelled the save operation
            return

        # The current state of the image is in self.pil_image
        # You might need to apply any additional transformations or overlays
        # that you want to be included in the saved image
        self.pil_image.save(file_path)


    def toggle_overlay(self):
        self.redraw_image()

    def adjust_overlay_opacity(self, event=None):
        self.overlay_opacity_entry.delete(0, tk.END)
        self.overlay_opacity_entry.insert(tk.END, f"{self.overlay_opacity_scale.get():.2f}")
        self.redraw_image()

    def adjust_suboverlay_opacity(self, event=None):
        self.redraw_image()

    def set_opacity_from_entry(self, event=None):
        self.master.focus()
        try:
            val = float(self.overlay_opacity_entry.get())
            if 0 <= val <= 1:
                self.overlay_opacity_scale.set(val)
                self.adjust_overlay_opacity()
        except ValueError:
            pass

    def set_max_min_from_entry(self, event=None):
        self.master.focus()
        try:
            val = float(self.min_value_entry.get())
            self.min_value = val
            val = float(self.max_value_entry.get())
            self.max_value = val
            self.set_image(self.image_list[self.image_index])
        except ValueError:
            pass

    def set_micron_factor(self, event=None):
        self.master.focus()
        try:
            val = float(self.micron_entry.get())
            self.micron_factor = val
            self.redraw_image()
        except ValueError:
            pass

    def pick_color(self):
        color = tkinter.colorchooser.askcolor(title="Choose Overlay Color")
        if color and color[1]:
            self.sub_overlay_colors[0] = color[1]

        self.toggle_color()
        self.toggle_color()
        self.redraw_image()

    def toggle_color(self, event=None):
        self.pencil_color = 'white' if self.pencil_color == 'black' else 'black'
        label_color = 'black' if self.pencil_color == 'black' else self.sub_overlay_colors[0]
        self.color_label.config(background=label_color) # Update the color preview$

    def on_resample_method_changed(self, selected_method):
        self.resample_method.set(selected_method)
        self.redraw_image()

    def calculate_image_range(self, radius, direction):
        if direction == "omi":
            start_index = max(0, self.image_index - radius)
            end_index = min(len(self.image_list), self.image_index + radius + 1)
        elif direction == "front":
            start_index = self.image_index
            end_index = min(len(self.image_list), self.image_index + radius + 1)
        elif direction == "back":
            start_index = max(0, self.image_index - radius)
            end_index = self.image_index + 1
        return start_index, end_index
    
    def enhance_image(self, image):
        if self.toggle_contrast_var.get():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
            image = clahe.apply(image.astype(np.uint8))
        return image
    
    def process_images(self):
        radius = int(self.radius_var.get())
        direction = self.direction_var.get()
        start_index, end_index = self.calculate_image_range(radius, direction)

        # Stack images as a 3D NumPy array
        images = np.stack([self.load_image(self.image_list[i], as_np=True) for i in tqdm(range(start_index, end_index))])
        if images.size > 0:
            if images.size == 1:
                result_image = images[0]
            else:
                operation = self.operation_var.get()
                if operation == "max":
                    result_image = np.max(images, axis=0)
                elif operation == "min":
                    result_image = np.min(images, axis=0)
                elif operation == "mean":
                    result_image = np.mean(images, axis=0)

            if self.min_value != 0 or self.max_value != 65535:
                result_image = (result_image - (self.min_value / 256.0)) * ( 65535.0 / (self.max_value - self.min_value))
                result_image = np.clip(result_image, 0, 255)
            result_image = result_image.astype(np.uint8)

            result_image = self.enhance_image(result_image)

            self.pil_image = Image.fromarray(result_image).convert("L")
            self.redraw_image()

    def set_image(self, filename):
        if not filename:
            return

        self.process_images()
        # self.draw_image(self.pil_image)

        self.master.title(self.my_title + " - " + os.path.basename(filename))
        self.label_image_info["text"] = f"{self.pil_image.format} : {self.pil_image.width} x {self.pil_image.height} {self.pil_image.mode}"
        os.chdir(os.path.dirname(filename))

    # Method to clear all SubOverlays
    def clear_suboverlays(self):
        self.sub_overlays = [self.sub_overlays[0]]
        self.sub_overlay_names = [self.sub_overlay_names[0]]
        self.redraw_image()
        self.update_suboverlay_dropdown()

    def operation_changed(self, _):
        self.process_images()

    def reset_to_middle_image(self, event=None):
        self.image_index = len(self.image_list) // 2
        self.set_image(self.image_list[self.image_index])

    def set_layer_from_entry(self, event=None):
        try:
            layer_index = int(self.layer_index_var.get())
            if 0 <= layer_index < len(self.image_list):
                self.image_index = layer_index
                self.set_image(self.image_list[self.image_index])
            else:
                tk.messagebox.showerror("Error", "Layer index out of range.")
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid layer index.")
        self.master.focus()  # Shift focus away from the entry field

    def show_previous_image(self, event, image_offset=1):
        if self.image_index - image_offset > 0:
            self.image_index -= image_offset
            self.set_image(self.image_list[self.image_index])

    def show_next_image(self, event, image_offset=1):
        if self.image_index < len(self.image_list) - image_offset:
            self.image_index += image_offset
            self.set_image(self.image_list[self.image_index])

    def generate_line(self, event):
        draw = ImageDraw.Draw(self.overlay_image)
        old_point = tuple(self.to_image_point(self.__old_event.x, self.__old_event.y)[:2])
        new_point = tuple(self.to_image_point(event.x, event.y)[:2])
        width = self.size_scale.get()
        draw.line([old_point, new_point], fill=self.pencil_color, width=width, joint='curve')
        # Draw circle with radius with/2
        draw.ellipse([old_point[0]-width/2, old_point[1]-width/2, old_point[0]+width/2, old_point[1]+width/2], fill=self.pencil_color)
        draw.ellipse([new_point[0]-width/2, new_point[1]-width/2, new_point[0]+width/2, new_point[1]+width/2], fill=self.pencil_color)

        self.redraw_image()

    def mouse_down_left(self, event):
        self.__old_event = event
        if self.shift_pressed and self.overlay_image:
            self.generate_line(event)
        self.mouse_is_pressed = True  # Mouse button pressed

    def mouse_up_left(self, event):  
        self.mouse_is_pressed = False  # Mouse button released

    def mouse_down_right(self, event):
        self.__old_event = event
        if self.overlay_image:
            self.generate_line(event)
        self.mouse_is_pressed_right = True  # Mouse button pressed

    def mouse_up_right(self, event):  
        self.mouse_is_pressed_right = False  # Mouse button released

    def mouse_move_left(self, event):
        if (self.pil_image == None) or (not self.mouse_is_pressed): # Check if mouse button is pressed
            return

        # Check if shift is pressed and mouse is dragged to draw
        if self.shift_pressed and self.mouse_is_pressed and self.overlay_image:
            self.generate_line(event)
        else: # Else case for dragging
            self.translate(event.x - self.__old_event.x, event.y - self.__old_event.y)

        self.redraw_image()
        self.__old_event = event

    def mouse_move_right(self, event):
        if (self.pil_image == None) or (not self.mouse_is_pressed_right): # Check if mouse button is pressed
            return

        if self.overlay_image:
            self.generate_line(event)

        self.redraw_image()
        self.__old_event = event

    def mouse_move(self, event):
        # Remove the old circle if it exists
        if self.cursor_circle:
            self.canvas.delete(self.cursor_circle)
        
        # Draw the new circle
        x, y = event.x, event.y 

        r = (self.size_scale.get() * self.global_scale_factor)// 2  # radius of circle
        self.cursor_circle = self.canvas.create_oval(x-r, y-r, x+r, y+r, outline=self.pencil_color)

        if (self.pil_image == None):
            return
        
        image_point = self.to_image_point(event.x, event.y)
        if image_point != []:
            uv_point = np.array([image_point[0] / self.pil_image.width, 1.0 - image_point[1] / self.pil_image.height])
            point_3d = find_uv_triangle(self.mesh_vertices, uv_point, self.kd_tree, self.triangle_data)
            if point_3d is None:
                point_3d = ["--", "--", "--"]
            else:
                point_3d = [f"{point_3d[0]:.2f}", f"{point_3d[1]:.2f}", f"{point_3d[2]:.2f}"]
            self.label_image_pixel["text"] = (f"2D: (x: {image_point[0]:.2f}, y: {image_point[1]}) 3D: (x: {point_3d[0]}, y: {point_3d[1]}, z: {point_3d[2]})")
        else:
            self.label_image_pixel["text"] = ("2D: (x: --, y: --) 3D: (x: --, y: --, z: --)")

    def mouse_leave_canvas(self, event):
        if self.cursor_circle:
            self.canvas.delete(self.cursor_circle)
            self.cursor_circle = None


    def mouse_double_click_left(self, event):
        if self.pil_image == None:
            return
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.redraw_image()
        self.reset_to_middle_image()

    def shift_press(self, event):
        self.shift_pressed = True

    def shift_release(self, event):
        self.shift_pressed = False

    def mouse_wheel(self, event):
        if self.pil_image == None:
            return

        if event.num == 5 or event.delta < 0:
            scale_factor = 0.8 if not self.shift_pressed else -5
        else: # event.num == 4 or event.delta > 0
            scale_factor = 1.25 if not self.shift_pressed else 5

        if not self.shift_pressed: 
            self.scale_at(scale_factor, event.x, event.y)
        else:
            self.rotate_at(scale_factor, event.x, event.y)
        self.redraw_image() # redraw the image

    def reset_transform(self):
        self.mat_affine = np.eye(3) 

    def translate(self, offset_x, offset_y):
        mat = np.eye(3)
        mat[0, 2] = float(offset_x)
        mat[1, 2] = float(offset_y)

        self.mat_affine = np.dot(mat, self.mat_affine)

    def scale(self, scale:float):
        mat = np.eye(3)
        mat[0, 0] = scale
        mat[1, 1] = scale

        self.mat_affine = np.dot(mat, self.mat_affine)
        scale_x = (self.mat_affine[0, 0]**2 + self.mat_affine[0, 1]**2)**0.5
        scale_y = (self.mat_affine[1, 0]**2 + self.mat_affine[1, 1]**2)**0.5
        self.global_scale_factor = scale_x
        # Correct small errors in the affine matrix
        scale_y = scale_x
        self.mat_affine[1, 1] = self.mat_affine[0, 0]
        self.mat_affine[1, 0] = -self.mat_affine[0, 1]
        # assert self.mat_affine[0, 0] == self.mat_affine[1, 1], f"Non-uniform scaling is not supported on affine (0,0) vs (1,1) {self.mat_affine[0, 0]} != {self.mat_affine[1, 1]}"
        # assert self.mat_affine[0, 1] == -self.mat_affine[1, 0], f"Non-uniform scaling is not supported on affine (0,1) -(1,0) {self.mat_affine[0, 1]} != {self.mat_affine[1, 0]}"
        # assert scale_x == scale_y, f"Non-uniform scaling is not supported on scale {scale_x} != {scale_y}"

    def scale_at(self, scale:float, cx:float, cy:float):
        self.translate(-cx, -cy)
        self.scale(scale)
        self.translate(cx, cy)

    def rotate(self, deg:float):
        mat = np.eye(3) 
        mat[0, 0] = math.cos(math.pi * deg / 180)
        mat[1, 0] = math.sin(math.pi * deg / 180)
        mat[0, 1] = -mat[1, 0]
        mat[1, 1] = mat[0, 0]

        self.mat_affine = np.dot(mat, self.mat_affine)

    def rotate_at(self, deg:float, cx:float, cy:float):

        self.translate(-cx, -cy)
        self.rotate(deg)
        self.translate(cx, cy)

    def zoom_fit(self, image_width, image_height):

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if (image_width * image_height <= 0) or (canvas_width * canvas_height <= 0):
            return

        self.reset_transform()

        scale = 1.0
        offsetx = 0.0
        offsety = 0.0

        if (canvas_width * image_height) > (image_width * canvas_height):
            scale = canvas_height / image_height
            offsetx = (canvas_width - image_width * scale) / 2
        else:
            scale = canvas_width / image_width
            offsety = (canvas_height - image_height * scale) / 2

        self.scale(scale)
        self.translate(offsetx, offsety)
    
    def to_image_point_unchecked(self, x, y):
        mat_inv = np.linalg.inv(self.mat_affine)
        image_point = np.dot(mat_inv, (x, y, 1.))
        return image_point
    
    def to_image_point(self, x, y):
        if self.pil_image == None:
            return []
        image_point = self.to_image_point_unchecked(x, y)
        if  image_point[0] < 0 or image_point[1] < 0 or image_point[0] > self.pil_image.width or image_point[1] > self.pil_image.height:
            return []

        return image_point
    
    def create_ruler(self, img_width, img_height, width, height, unit_size, min_unit_length=15):
        """
        Create a ruler image with the specified width, height, and unit size.
        """
        unit_size = int(unit_size)
        unit_size = max(1, unit_size)  # Ensure unit size is at least 1
        ruler = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(ruler)

        unit_range = range(height, width, unit_size)
        if len(unit_range) < min_unit_length:
            unit_range = [i for i in range(height, height + min_unit_length*unit_size, unit_size)]

        # Draw ruler lines and numbers
        for i in unit_range:
            line_height = height // 2 if (i - height) % (5 * unit_size) else height
            draw.line([(i + height, 0), (i + height, line_height)], fill="white", width=1)
            draw.text((i + height, line_height), str((i - height) // unit_size), fill="white")

        # Draw ruler lines and numbers
        for i in unit_range:
            line_height = height // 2 if (i - height) % (5 * unit_size) else height
            draw.line([(0, i + height), (line_height, i + height)], fill="white", width=1)
            draw.text((line_height, i + height), str((i - height) // unit_size), fill="white")

        return ruler
    
    def update_threshold_value(self, val):
        self.ff_threshold = int(float(val))
        self.bucket_threshold_var.set(f"{self.ff_threshold}")
        print(self.ff_threshold)

    def update_max_propagation(self, val):
        self.max_propagation_steps = int(float(val))
        self.max_propagation_var.set(f"{self.max_propagation_steps}")

    def threaded_flood_fill(self, event):
        if self.flood_fill_active:
            return
        self.flood_fill_active = True
        click_coordinates = self.to_image_point(event.x, event.y)[:2]
        click_coordinates[0] = int(click_coordinates[0])
        click_coordinates[1] = int(click_coordinates[1])
        click_coordinates = tuple(click_coordinates)
        # Run flood_fill_3d in a separate thread
        thread = threading.Thread(target=self.flood_fill_2d, args=(click_coordinates,))
        thread.start()

    def flood_fill_2d(self, start_coord):
        pil_image = self.pil_image
        queue = deque([start_coord])
        target_color = int(pil_image.getpixel(start_coord))
        visited = set()
        counter = 0
        if self.overlay_image.mode == 'RGB':
            # Convert to a tuple of integers for RGB
            value = (int(255), int(255), int(255))
        else:
            value = int(255)
        while self.flood_fill_active and queue and counter < self.max_propagation_steps:
            cx, cy = queue.popleft()

            if (cx, cy) in visited or not (0 <= cx < pil_image.width and 0 <= cy < pil_image.height):
                continue

            visited.add((cx, cy))


            pixel_value = int(pil_image.getpixel((cx,cy)))

            if abs(pixel_value - target_color) <= self.ff_threshold:
                try:
                    self.overlay_image.putpixel((int(cx), int(cy)), value)
                except TypeError as e:
                    print(f"Error: {e}, Coordinates: ({cx}, {cy}), Value: {value}, Mode: {self.overlay_image.mode}")
                counter += 1
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        queue.append((cx + dx, cy + dy))

            if counter % 10 == 0:
                self.redraw_image()

        if self.flood_fill_active == True:
            self.flood_fill_active = False
            self.redraw_image()

    def draw_image(self, pil_image):
        if pil_image == None:
            return

        self.pil_image = pil_image

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        mat_inv = np.linalg.inv(self.mat_affine)

        affine_inv = (
            mat_inv[0, 0], mat_inv[0, 1], mat_inv[0, 2],
            mat_inv[1, 0], mat_inv[1, 1], mat_inv[1, 2]
            )

        dst = self.pil_image.transform(
                    (canvas_width, canvas_height),
                    Image.Transform.AFFINE,   
                    affine_inv,   
                    self.resampling_methods[self.resample_method.get()]
                    )
        
        if dst.mode != 'RGBA':
            dst = dst.convert('RGBA')

        if self.overlay_visibility.get():
            # Overlaying SubOverlays
            for i, sub_overlay in enumerate(self.sub_overlays):
                if i == 0: continue # skip overlay image

                sub_overlay_transformed = sub_overlay.transform(
                    (canvas_width, canvas_height),
                    Image.Transform.AFFINE,
                    affine_inv,
                    self.resampling_methods[self.resample_method.get()]
                )
                
                # Ensure the image is RGBA (has an alpha channel)
                if sub_overlay_transformed.mode != 'RGBA':
                    sub_overlay_transformed = sub_overlay_transformed.convert('RGBA')

                r, g, b, a = sub_overlay_transformed.split()
                grayscale = sub_overlay_transformed.convert("L")
                # Scale the RGB values based on the brightness slider
                grayscale = ImageEnhance.Brightness(grayscale).enhance(self.suboverlay_brightness_scale.get())
                alpha = grayscale  # Use grayscale directly, without inverting
                sub_overlay_color = Image.new('RGB', sub_overlay_transformed.size, self.sub_overlay_colors[i])
                final_sub_overlay = Image.composite(sub_overlay_color, sub_overlay_transformed, grayscale)
                final_sub_overlay.putalpha(alpha)


                # Adjust the opacity based on the value of the slider
                alpha = ImageEnhance.Brightness(alpha).enhance(self.suboverlay_opacity_scale.get())
                final_sub_overlay.putalpha(alpha)

                dst.paste(final_sub_overlay, (0, 0), final_sub_overlay)

        # Overlaying the additional PNG
        if self.overlay_visibility.get() and self.overlay_image:
            overlay_transformed = self.overlay_image.transform(
                (canvas_width, canvas_height),
                Image.Transform.AFFINE,
                affine_inv,
                self.resampling_methods[self.resample_method.get()]
            )
            
            # Ensure the image is RGBA (has an alpha channel)
            if overlay_transformed.mode != 'RGBA':
                overlay_transformed = overlay_transformed.convert('RGBA')

            r, g, b, a = overlay_transformed.split()
            grayscale = overlay_transformed.convert("L")
            alpha = grayscale  # Use grayscale directly, without inverting
            yellow = Image.new('RGB', overlay_transformed.size, self.sub_overlay_colors[0])
            final_overlay = Image.composite(yellow, overlay_transformed, grayscale)
            final_overlay.putalpha(alpha)

            # Adjust the opacity based on the value of the slider
            alpha = ImageEnhance.Brightness(alpha).enhance(self.overlay_opacity_scale.get())
            final_overlay.putalpha(alpha)

            dst.paste(final_overlay, (0, 0), final_overlay)

        # Add a ruler to the bottom right of the image
        ruler_width, ruler_height = 500, 100  # Customize as needed
        unit_size = self.global_scale_factor * (1.0 / self.micron_factor)  # Customize the unit size for the ruler
        image_width, image_height = dst.size
        ruler = self.create_ruler(image_width, image_height, ruler_width, ruler_height, unit_size)

        # Calculate position for the ruler (bottom right)
        ruler_position = (0, 0)

        # Paste the ruler onto the image
        dst.paste(ruler, ruler_position, ruler)

        im = ImageTk.PhotoImage(image=dst)

        item = self.canvas.create_image(
                0, 0, 
                anchor='nw',
                image=im  
                )

        self.image = im
        # Update the layer index display
        self.layer_index_var.set(str(self.image_index))

    def redraw_image(self):
        if self.pil_image == None:
            return
        self.draw_image(self.pil_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
