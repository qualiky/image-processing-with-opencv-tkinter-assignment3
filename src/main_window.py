import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from image_processor import ImageProcessor
from typing import Callable, Optional

# PLEASE REFER TO THE REFERENCE UI SCREENSHOT BEFORE YOU WORK ON YOUR SECTIONS

# SET 1 - TBD: Sandeep
class ImageProcessorApp:
    """This is the main application class, initialised by main"""

    def __init__(self):
        """Initialisation happens here."""
        # Things to be done:
        # 0. Create application root window & initialise the image processor
        # 1. Create widgets from Tkinter
        # 2. Setup the layout and add widgets to layout
        # 3. Update the display window with the layouts

    def _create_widgets(self) -> None:
        """All GUI components need to be created here"""
        # Things to be done:
        # 1. Create control panel
        # 2. Create image canvas
        # 3. Create menu manager
        # 4. Add these widgets to main root

    def _setup_layout(self) -> None:
        """Setup the layout here"""
        # Things to be done:
        # 1. ControlPanel constructor to pack the control to left
        # 2. Pack the canvas to fill the rest of the space

    def _update_display(self) -> None:
        """Update the image display and status bar"""
        # Things to be done:
        # 1. Get current image
        # 2. Display filename, dimension, format as needed

    def run(self) -> None:
        """Run the application"""
        # self.root.mainloop() goes here


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
    """Status bar component for displaying image metadata"""

    def __init__(self, parent: tk.Widget):
        """Initialise the status bar

        Args:
            parent: Parent tkinter widget
        """

    def update_metadata(self, filename: str, dimensions: tuple, format_name: str) -> None:
        """Update the status bar metadata of the image IF the image changes

        Args:
            filename: Image filename,
            dimensions: Tuple(width: int, height: int)
            format_name: Image format

        """


# Set 4 - TBD: Yasmeen
class MenuManager:
    """Menu bar management."""

    def __init__(
        self, root: tk.Tk, processor: ImageProcessor, update_callback: Callable
    ):
        """
        Initialize menu manager on the top of the screen

        Args:
            root: Root window
            processor: ImageProcessor instance
            update_callback: Callback to update display. A function argument is needed here
        """

    def _create_file_menu(self) -> None:
        """Create file menu."""

    def _create_edit_menu(self) -> None:
        """Create edit menu."""

    def _open_file(self) -> None:
        """Open file dialog."""

    def _save_file(self) -> None:
        """Save current image on the same base file"""

    def _save_as_file(self) -> None:
        """Save current image with new filename"""

    def _undo(self) -> None:
        """Undo last operation"""

    def _redo(self) -> None:
        """Redo last undone operation"""


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

    def _create_basic_controls(self) -> None:
        """Create basic filter controls"""

    def _create_adjustment_controls(self) -> None:
        """Create adjustment sliders"""

    def _create_transform_controls(self) -> None:
        """Create transformation controls"""

    def _create_resize_controls(self) -> None:
        """Create resize controls"""

    def _apply_grayscale(self) -> None:
        """Apply grayscale filter"""

    def _apply_edge_detection(self) -> None:
        """Apply edge detection filter"""

    def _rotate_image(self, angle: int) -> None:
        """Rotate image by specified angle"""

    def _flip_image(self, direction: str) -> None:
        """Flip image in specified direction"""

    def _resize_image(self) -> None:
        """Resize image to specified dimensions"""

    def _start_blur_adjustment(self) -> None:
        """Start blur adjustment, save base state only if not already adjusting blur."""

    def _on_blur_preview(self) -> None:
        """Preview blur while dragging."""

    def _finish_blur_adjustment(self) -> None:
        """Finish blur adjustment - commit to history, BUTTTTTT keep the base for further adjustments"""

    def _start_brightness_adjustment(self) -> None:
        """Start brightness adjustment - save base state"""

    def _on_brightness_preview(self) -> None:
        """Preview brightness while dragging"""

    def _finish_brightness_adjustment(self) -> None:
        """Finish brightness adjustment - commit to history"""

    def _start_contrast_adjustment(self) -> None:
        """Start contrast adjustment - save base state"""

    def _on_contrast_preview(self) -> None:
        """Preview contrast while dragging"""

    def _finish_contrast_adjustment(self) -> None:
        """Finish contrast adjustment - commit to history"""

    def _reset_image(self) -> None:
        """Reset image to original state"""

if __name__ == "__main__":
    app = ImageProcessorApp()
    app.run()
