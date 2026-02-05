import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional, Callable

from .image_processor import ImageProcessor

# PLEASE REFER TO THE REFERENCE UI SCREENSHOT BEFORE YOU WORK ON YOUR SECTIONS


# SET 2 - TBD: Aryan
class ImageCanvas:
    """Image canvas component that displays the image canvas"""

    def __init__(self, parent: tk.Widget, width: int = 800, height: int = 600):
        """Initialise the image canvas where image is loaded & worked on

        Args:
            parent: parent container Widget
            width: canvas width
            height: canvas height

        """
        self._parent = parent
        self._width = width
        self._height = height
        # just a simple canvas to draw the image on
        self._canvas = tk.Canvas(parent, width=width, height=height)
        self._image_id: Optional[int] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None
        self._display_scale: float = 1.0

    def display_image(self, image: np.ndarray) -> None:
        """Show an image on the canvas"""

        if image is None:
            return

        try:
            # OpenCV uses BGR, PIL wants RGB
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image[:, :, ::-1])

            img_w, img_h = pil_image.size
            if img_w == 0 or img_h == 0:
                return

            # figure out how much to shrink to fit
            scale_w = self._width / img_w
            scale_h = self._height / img_h
            self._display_scale = min(scale_w, scale_h, 1.0)

            # only resize if it's too big
            if self._display_scale < 1:
                new_w = int(img_w * self._display_scale)
                new_h = int(img_h * self._display_scale)
                pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

            # draw it in the center
            self._tk_image = ImageTk.PhotoImage(pil_image)
            self._canvas.delete("all")
            self._image_id = self._canvas.create_image(
                self._width // 2, self._height // 2,
                image=self._tk_image, anchor="center"
            )

        except:
            print("Error displaying image")

    def clear_canvas(self) -> None:
        """Remove everything from the canvas"""
        self._canvas.delete("all")
        self._image_id = None
        self._tk_image = None
        self._display_scale = 1.0

    def get_display_scale(self) -> float:
        """Return current scale factor"""
        return self._display_scale



# Set 3 - TBD: Bishesh
class StatusBar:
    """Status bar component for displaying image information."""

    def __init__(self, parent: tk.Widget):
        """
        Initialize status bar.

        Args:
            parent: Parent widget
        """
        self.frame = ttk.Frame(parent)  # Create the main frame for the status bar
        self.frame.pack(fill=tk.X, side=tk.BOTTOM)  # Pack at the bottom of the parent, filling horizontally

        self.filename_label = ttk.Label(self.frame, text="No image loaded")  # Display filename
        self.filename_label.pack(side=tk.LEFT, padx=5)  # Filename positioning with padding

        self.separator = ttk.Separator(self.frame, orient=tk.VERTICAL)  # Seperator between filename and dimensions
        self.separator.pack(side=tk.LEFT, fill=tk.Y, padx=5)  # Fill with padding on both sides

        self.dimensions_label = ttk.Label(self.frame, text="Dimensions: -")  # Display image dimensions (width x height)
        self.dimensions_label.pack(side=tk.LEFT, padx=5)

        self.separator2 = ttk.Separator(self.frame, orient=tk.VERTICAL)  # Seperator between dimensions and format
        self.separator2.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.format_label = ttk.Label(self.frame, text="Format: -")  # Display the image format
        self.format_label.pack(side=tk.LEFT, padx=5)

    def update_info(self, filename: str, dimensions: tuple, format_name: str) -> None:
        """
        Update status bar information.

        Args:
            filename: Image filename
            dimensions: Image dimensions (width, height)
            format_name: Image format
        """
        self.filename_label.config(text=f"File: {filename}")  # Update the filename display
        # Update dimensions in "widthxheight" format
        # dimension[0]=width, dimension[1]=height
        self.dimensions_label.config(
            text=f"Dimensions: {dimensions[0]}x{dimensions[1]}"
        )
        self.format_label.config(text=f"Format: {format_name}")  # Update the format display


# Set 4 - TBD: Yasmeen
class MenuManager:
    """Menu bar management."""
    def __init__(self, root: tk.Tk, processor: ImageProcessor, update_callback: Callable):
        self.root = root
        self.processor = processor
        self.update_callback = update_callback

        self.menubar = tk.Menu(root)
        root.config(menu=self.menubar)

        self._create_file_menu()
        self._create_edit_menu()
        """
        Initialize menu manager on the top of the screen

        Args:
            root: Root window
            processor: ImageProcessor instance
            update_callback: Callback to update display. A function argument is needed here
        """

    def _create_file_menu(self) -> None:
        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self._open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self._save_file)
        file_menu.add_command(label="Save As", command=self._save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        self.menubar.add_cascade(label="File", menu=file_menu)

    def _create_edit_menu(self) -> None:
        edit_menu = tk.Menu(self.menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self._undo)
        edit_menu.add_command(label="Redo", command=self._redo)

        self.menubar.add_cascade(label="Edit", menu=edit_menu)

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("Image Files", "*.png *.jpg *.bmp")
            ]
        )
        if not path:
            return

        try:
            self.processor.load_image(path)
            self.update_callback()
        except Exception as e:
            messagebox.showerror("Open Error", str(e))

    def _save_file(self) -> None:
        try:
            if not self.processor.current_path:
                self._save_as_file()
            else:
                self.processor.save_image()
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _save_as_file(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp")
            ],
        )
        if not path:
            return

        try:
            self.processor.save_image(path)
        except Exception as e:
            messagebox.showerror("Save As Error", str(e))

    def _undo(self) -> None:
        try:
            self.processor.undo()
            self.update_callback()
        except Exception as e:
            messagebox.showinfo("Undo", str(e))

    def _redo(self) -> None:
        try:
            self.processor.redo()
            self.update_callback()
        except Exception as e:
            messagebox.showinfo("Redo", str(e))


# Set 5 - TBD: Sandeep
class ControlPanel:
    """Control panel for filters and effects."""

    def __init__(
        self, parent: tk.Widget, processor: ImageProcessor, update_callback: Callable
    ):
        """
        Initialize control panel.

        Args:
            parent: Parent widget
            processor: ImageProcessor instance
            update_callback: Callback to update display
        """
        self.parent = parent
        self.processor = processor
        self.update_callback = update_callback

        self.frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        # Take up the entire vertical space on the left
        self.frame.pack(fill=tk.Y, side=tk.LEFT, padx=5, pady=5)

        # Tracking active slider, to know which transformation to apply. Without this, granular atomic update is a massive pain in the neck
        self._slider_active = None

        # Now, sub-widgets created by individual functions:
        # 1. Basic controls: Basic grayscale & edge detection (buttons)
        self._create_basic_controls()
        # 2. Adjustment controls: Blur, Brightness, Contrast (sliders)
        self._create_adjustment_controls()
        # 3. Transform controls: Rotate 90, 180, 270, Flip horizontal & flip vertical
        self._create_transform_controls()
        # 4. Resize controls: Image resize with w, h, apply button, and restore original image
        self._create_resize_controls()



    def _create_basic_controls(self) -> None:
        """Create basic filter controls"""
        ttk.Label(self.frame, text="Basic Filters", font=("Helvetica", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=5)

        # Convert to Grayscale button
        ttk.Button(self.frame, text="Convert to Grayscale", command=self._apply_grayscale).grid(row=1, column=0, columnspan=2, pady=2, sticky="ew")

        # Edge Detection button
        ttk.Button(self.frame, text="Edge Detection", command=self._apply_edge_detection).grid(row=2, column=0, columnspan=2, pady=2, sticky="ew")



    def _create_adjustment_controls(self) -> None:
        """Create adjustment sliders"""
        ttk.Label(self.frame, text="Adjustments", font=("Helvetica", 10, "bold")).grid(row=3, column=0, columnspan=2, pady=(10,5))

        # Now comes the difficult part - three sliders, with own labels. Need to be careful with how it is laid out, and how the mouse buttons bind with the functions

        # Slider 1: Blur
        ttk.Label(self.frame, text="Blur:").grid(row=4, column=0, sticky="w")

        # we need a "state" variable to save the value of changes. Similar to useState in React. this intvar automatically observes and updates itself on slider change
        self.blur_var = tk.IntVar(value=0)

        # Now, the slider
        blur_scale = ttk.Scale(self.frame, from_=0, to=20, variable=self.blur_var, orient=tk.HORIZONTAL)
        blur_scale.grid(row=4, column=1, sticky="ew")

        # On press of left mouse button, we need to call the function that starts blur adjustment
        blur_scale.bind("<Button-1>", lambda e: self._start_blur_adjustment())

        # As the slider moves, the blur needs to be preview-able live
        blur_scale.bind("<B1-Motion>", lambda e: self._on_blur_preview())

        # On release of the pressed left mouse button, we need to call function that finishes blur adjustment and pushes it to the history stack
        blur_scale.bind("<ButtonRelease-1>", lambda e: self._finish_blur_adjustment())


        # Slider 2: Brightness
        ttk.Label(self.frame, text="Brightness:").grid(row=5, column=0, sticky="w")

        # we need a "state" variable to save the value of changes as we did earlier with blur. Similar to useState in React. this intvar automatically observes and updates itself on slider change
        self.brightness_var = tk.IntVar(value=0)

        # Now, the slider
        brightness_scale = ttk.Scale(self.frame, from_=-100, to=100, variable=self.brightness_var, orient=tk.HORIZONTAL)
        brightness_scale.grid(row=5, column=1, sticky="ew")

        # On press of left mouse button, we need to call the function that starts brightness adjustment
        brightness_scale.bind("<Button-1>", lambda e: self._start_brightness_adjustment())

        # As the slider moves, the brightness needs to be preview-able live
        brightness_scale.bind("<B1-Motion>", lambda e: self._on_brightness_preview())

        # On release of the pressed left mouse button, we need to call function that finishes brightness adjustment and pushes it to the history stack
        brightness_scale.bind("<ButtonRelease-1>", lambda e: self._finish_brightness_adjustment())

        # Slider 3: Contrast
        ttk.Label(self.frame, text="Contrast:").grid(row=6, column=0, sticky="w")

        # we need a "state" variable to save the value of changes as we did earlier with brightness. Similar to useState in React. this intvar automatically observes and updates itself on slider change
        self.contrast_var = tk.IntVar(value=0)

        # Now, the slider
        contrast_scale = ttk.Scale(self.frame, from_=0.1, to=3.0, variable=self.contrast_var, orient=tk.HORIZONTAL)
        contrast_scale.grid(row=6, column=1, sticky="ew")

        # On press of left mouse button, we need to call the function that starts contrast adjustment
        contrast_scale.bind("<Button-1>", lambda e: self._start_contrast_adjustment())

        # As the slider moves, the contrast needs to be preview-able live
        contrast_scale.bind("<B1-Motion>", lambda e: self._on_contrast_preview())

        # On release of the pressed left mouse button, we need to call function that finishes contrast adjustment and pushes it to the history stack
        contrast_scale.bind("<ButtonRelease-1>", lambda e: self._finish_contrast_adjustment())



    def _create_transform_controls(self) -> None:
        """Create transformation controls"""
        ttk.Label(self.frame, text="Transform", font=("Helvetica", 10, "bold")).grid(
            row=7, column=0, columnspan=2, pady=(10, 5)
        )
        # The buttons will be in a grid
        transform_frame = ttk.Frame(self.frame)
        transform_frame.grid(row=8, column=0, columnspan=2, pady=2)

        # Rotate 90 button
        ttk.Button(
            transform_frame, text="Rotate 90°", command=lambda: self._rotate_image(90)
        ).pack(side=tk.LEFT, padx=2)

        # Rotate 180 button
        ttk.Button(
            transform_frame, text="Rotate 180°", command=lambda: self._rotate_image(180)
        ).pack(side=tk.LEFT, padx=2)

        # Rotate 270 button
        ttk.Button(
            transform_frame, text="Rotate 270°", command=lambda: self._rotate_image(270)
        ).pack(side=tk.LEFT, padx=2)

        # Separate frames for flip buttons
        flip_frame = ttk.Frame(self.frame)
        flip_frame.grid(row=9, column=0, columnspan=2, pady=2)

        # Flip Horizontal button
        ttk.Button(
            flip_frame, text="Flip H", command=lambda: self._flip_image("horizontal")
        ).pack(side=tk.LEFT, padx=2)

        # Flip Vertical button
        ttk.Button(
            flip_frame, text="Flip V", command=lambda: self._flip_image("vertical")
        ).pack(side=tk.LEFT, padx=2)

    def _create_resize_controls(self) -> None:
        """Create resize controls"""
        ttk.Label(self.frame, text="Resize", font=("Helvetica", 10, "bold")).grid(
            row=10, column=0, columnspan=2, pady=(10, 5)
        )

        size_frame = ttk.Frame(self.frame)
        size_frame.grid(row=11, column=0, columnspan=2, pady=2)

        # As done earlier, width and height variables need to be saved in a StringVar for reactive changes
        ttk.Label(size_frame, text="W:").pack(side=tk.LEFT)
        self.width_var = tk.StringVar(value="1000")
        ttk.Entry(size_frame, textvariable=self.width_var, width=6).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Label(size_frame, text="H:").pack(side=tk.LEFT)
        self.height_var = tk.StringVar(value="800")
        ttk.Entry(size_frame, textvariable=self.height_var, width=6).pack(
            side=tk.LEFT, padx=2
        )

        # Apply resize
        ttk.Button(self.frame, text="Apply Resize", command=self._resize_image).grid(
            row=12, column=0, columnspan=2, pady=2, sticky="ew"
        )

        # Revert to original
        ttk.Button(
            self.frame, text="Reset to Original", command=self._reset_image
        ).grid(row=13, column=0, columnspan=2, pady=(10, 2), sticky="ew")


    def _apply_grayscale(self) -> None:
        """Apply grayscale filter"""
        # First, we confirm and commit the preview
        self.processor.commit_preview(clear_base=True)
        # Then we reset the active slider
        self._slider_active = None
        # The actual grayscale application is called here
        self.processor.apply_grayscale()
        # The update callback now displays grayscale image
        self.update_callback()

    def _apply_edge_detection(self) -> None:
        """Apply edge detection filter."""
        self._slider_active = None
        self.processor.apply_edge_detection()
        self.update_callback()

    def _rotate_image(self, angle: int) -> None:
        """Rotate image by specified angle."""
        self._slider_active = None
        self.processor.rotate_image(angle)
        self.update_callback()

    def _flip_image(self, direction: str) -> None:
        """Flip image in specified direction."""
        self._slider_active = None
        self.processor.flip_image(direction)
        self.update_callback()

    def _resize_image(self) -> None:
        """Resize image to specified dimensions."""
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            if width > 0 and height > 0:
                self._slider_active = None
                self.processor.resize_image(width, height)
                self.update_callback()
        except ValueError:
            messagebox.showerror("Error", "Invalid dimensions entered")


    # THESE METHODS ARE CRUCIAL FOR CORRECT APPLICATION OF FILTERS
    def _start_blur_adjustment(self) -> None:
        """Start blur adjustment, save base state only if not already adjusting blur."""
        if self._slider_active != "blur":
            self.processor.start_preview()
        self._slider_active = "blur"

    def _on_blur_preview(self) -> None:
        """Preview blur while dragging."""
        if self._slider_active == "blur":
            intensity = self.blur_var.get()
            self.processor.apply_blur_preview(intensity)
            self.update_callback()

    def _finish_blur_adjustment(self) -> None:
        """Finish blur adjustment - commit to history, BUTTTTTT keep the base for further adjustments"""
        if self._slider_active == "blur":
            self.processor.commit_preview(clear_base=False)
            # self._slider_active = None

    def _start_brightness_adjustment(self) -> None:
        """Start brightness adjustment - save base state"""
        if self._slider_active != "brightness":
            self.processor.start_preview()
        self._slider_active = "brightness"

    def _on_brightness_preview(self) -> None:
        """Preview brightness while dragging"""
        if self._slider_active == "brightness":
            brightness = self.brightness_var.get()
            self.processor.adjust_brightness_preview(brightness)
            self.update_callback()

    def _finish_brightness_adjustment(self) -> None:
        """Finish brightness adjustment - commit to history"""
        if self._slider_active == "brightness":
            self.processor.commit_preview(clear_base=False)
            # self._slider_active = None

    def _start_contrast_adjustment(self) -> None:
        """Start contrast adjustment - save base state"""
        if self._slider_active != "contrast":
            self.processor.start_preview()
        self._slider_active = "contrast"

    def _on_contrast_preview(self) -> None:
        """Preview contrast while dragging"""
        if self._slider_active == "contrast":
            contrast = self.contrast_var.get()
            self.processor.adjust_contrast_preview(contrast)
            self.update_callback()

    def _finish_contrast_adjustment(self) -> None:
        """Finish contrast adjustment - commit to history"""
        if self._slider_active == "contrast":
            self.processor.commit_preview(clear_base=False)
            # self._slider_active = None

    def _reset_image(self) -> None:
        """Reset image to original state"""
        self._slider_active = None
        self.processor._preview_base = None
        self.processor.reset_to_original()
        self.update_callback()
        self.blur_var.set(0)
        self.brightness_var.set(0)
        self.contrast_var.set(1.0)


# SET 1 - TBD: Sandeep
class ImageProcessorApp:
    """This is the main application class, initialised by main"""

    def __init__(self):
        """Initialisation happens here."""
        self.root = tk.Tk()
        self.root.title("Image Processor")
        self.root.geometry("1000x800")
        self.root.minsize(800,600)

        self.processor = ImageProcessor()

        self._create_widgets()
        self._setup_layout()
        self._update_display()

    def _create_widgets(self) -> None:
        """All GUI components need to be created here"""
        # Things to be done:

        # 0. Create main container first, actually
        # IMPORTANT: ttk, NOT tk. ttk = Themed TK, perfect for native look/feel
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # 1. Create control panel
        self.control_panel = ControlPanel(self.main_container, self.processor, self._update_display)

        # 2. Create image canvas
        self.image_canvas = ImageCanvas(self.main_container)

        # 3. Create menu manager
        self.menu_manager = MenuManager(self.root, self.processor, self._update_display)

        # 4. Create Status bar
        self.status_bar = StatusBar(self.root)

        # 5. My bad I'm dumb - no need to add everything to root, since main_container is passed already to each widget function

    def _setup_layout(self) -> None:
        """Setup the layout here"""
        # Things to be done:
        # 1. ControlPanel constructor to pack the control to left - TBD: to be implemented in the ControlPanel constructor, NOT here
        # 2. Pack the canvas to fill the rest of the space
        self.image_canvas.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _update_display(self) -> None:
        """Update the image display and status bar"""
        # Things to be done:
        # 1. Get current image and set it to the current image in canvas, basically live rendering on changes
        current_image = self.processor.get_current_image()
        self.image_canvas.display_image(current_image)
        # 2. Display filename, dimension, format as needed
        filename, dimensions, format_name = self.processor.get_image_info()
        self.status_bar.update_info(filename, dimensions, format_name)

    def run(self) -> None:
        """Run the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = ImageProcessorApp()
    app.run()
