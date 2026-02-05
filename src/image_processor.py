import numpy as np
import cv2
from typing import Optional, Tuple, List
from cv2 import rotate, ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE, flip, resize, INTER_AREA, INTER_LINEAR

class ImageProcessor:
    """Core class that processes images with OpenCV. All image manipulation happen with an ImageProcessor instance"""

    def __init__(self):
        """Initialising ImageProcessor with default states"""

        # The original image is stored as an array of pixel data of m x n dimensions. An individual pixel is an (R, G, B) value. Optional if no image is chosen
        self._original_image: Optional[np.ndarray] = None

        # Current image is the pixel values after filters and transformations have been applied to the original image's pixel data. This is the value we export at the end when the image is exported. Optional because there may be no changes to the image
        self._current_image: Optional[np.ndarray] = None

        # Preview base is the temporary image where the preview of the applied changes are visible. In case of preview base, if the transformation is reversed, the preview base goes back to the last value of the current image. Especially needed with blur/contrast/brightness, because any operation on the current image with blur and contrast applies cumulative transformation to an already blurred or contrasted pixel.
        self._preview_base: Optional[np.ndarray] = None

        # The currently open file name + path. Optional since no files can be open during the initialisation of this function
        self._filename : Optional[str] = None

        # History is a stack where all transformations are stored once they've been applied. Whenever a transformation is applied, it is pushed to the stack. If a transformation is undone, the stack is popped. Basically, each transformation stores a snapshot of the current image's applied filters.
        self._history: List[np.ndarray] = []

        # History index is the current stack pointer. Initialises at the bottom of the empty stack, hence -1
        self._history_index: int = -1

        # Max history size. The more history, the more snapshots of the transformations. It's crucial we balance out the number of snapshots and the history size. Large image resolutions absolutely hog up the memory real quick. 20 is sensible for now.
        self._max_history: int = 20



    # Functions to be implemented:
    # SET 1: IMAGE LOAD AND SAVE OPERATIONS. TBD: Bishesh
    def load_image(self, filepath: str) -> bool:
        """Load the image from a given path.

        Args:
            filepath: Path to image file. Operating system agnostic, works for both UNIX-like systems and Windows. Returned by the file picker.

        Returns:
            bool: If successful, returns True, else False.

        """
        print(f"This function loads file from {filepath}")



    def save_image(self, filepath: str) -> bool:
        """Save the current image snapshot as a file.

        Args:
            filepath: Path where file needs to be saved. Overwriting, same-name files, and conflicts are handled by the OS filepicker (tested on Finder, works with no additional requirements.)

        Returns:
            bool: If successful, returns True, else False.

        """
        print(f"This function saves image to {filepath}")


    def get_image_metadata(self) -> Tuple[str, Tuple[int, int], str]:
        """Load metadata from the currently selected image

        Returns:
            Tuple: filename as string, image resolution as a Tuple[int, int], format as string

        """
        print(f"This function returns metadata of the image at path {self._filename}")


    def reset_image(self) -> None:
        """Reset the currently transformed image to the original state"""
        print(f"This function resets image back to original state at {self._filename}")

    # SET 2: Primary Image Transformations. TBD: Sandeep.

    def apply_grayscale_filter(self) -> None:
        """Convert the current image to grayscale image"""

        if self._current_image is not None:
            # The output of this first function is a grayscale image. This is basically doing [B=128, G=128, R=128] to a [GRAY=80] conversion
            gray = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2GRAY)

            # We CANNOT use this output. We need to convert this consistent, single channel gray image back to a 3-channel BGR image, since everything we do is in BGR. The following line is equivalent of running [GRAY=80] to [B=80, G=80, R=80]
            self._current_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            self.add_to_history()



    def apply_blur_filter(self, intensity: int) -> None:
        """Apply gaussian blur to an image with adjustable intensity. This intensity is in the form of int value.

        We only accept values between 0 to 20 in our implementation. This is because gaussian blur is a weighted average of pixels, and the kernel size above the pixels that uses weighted average to calculate blur grows with `intensity * 2 + 1` - effectively, with value 100, the function would have to perform weighted average of the center pixel with a 201 x 201 sized kernel, making it computationally expensive.
        """
        if self._current_image is not None:
            if intensity > 0:
                # The blur kernel resolution has to be defined manually based on the size of intensity
                kernel_size = 2 * intensity + 1
                # The third parameter is sigmaX, which ranges between 0.0 to 2.0. With a smaller sigma (or standard deviation) value, the central pixel has heavier weight - which results in minimal blur. With a bigger deviation value, the surrounding pixels have a closer distribution, making the blur heavy
                # With a small kernel weight:
                # [0.001  0.01   0.001]
                # [0.01   0.95   0.01 ]  (Center heavily weighted)
                # [0.001  0.01   0.001]
                # With a bigger kernel weight:
                # Kernel weights:
                # [0.05   0.10   0.05]
                # [0.10   0.40   0.10]    (More even distribution)
                # [0.05   0.10   0.05]
                self._current_image = cv2.GaussianBlur(self._current_image, (kernel_size, kernel_size), 0)
            self.add_to_history()


    def apply_edge_detection(self, low_threshold: int = 50, high_threshold: int = 150) -> None:
        """Applies Canny edge detection to detect the edges of an image.

        Args:
            low_threshold: Lower threshold value of the gradient for edge detection, anything below this is 100% not an edge
            high_threshold: Upper threshold value of the gradient for edge detection, anything above this is 100% an edge

        Under the hood, Canny edge detection uses 5-step process to detect sharp intensity change, crucial for edge detection:

        1. Gaussian blur for noise reduction
        2. Gradient calculation using Sobel filter to calculate how fast pixel values change in X and Y directions. Fast changes = likely an edge
        3. Non-maximum suppressions to create thin edges instead of thick blobs. Which means, only the mode values are saved as edges, the rest is discarded.
        4. Double thresholding, where gradient magnitude below the low_threshold is 100% NOT an edge, while gradient magnitude above the high_threshold is 100% an edge. Any value between low and high could be an edge, depending on the connectivity.
        5. Edge tracking by hysteresis, where only strongly connected edges are saved. The rest is discarded, preventing isolated noise appearing as edges.

        Good thing is OpenCV provides an implementation for this directly, so we don't have to implement this in 5 steps. However, we need to provide the low and high threshold manually for this function. 50 and 150 are sensible defaults for this.
        """
        if self._current_image is not None:
            # Same stuff as grayscale filtering - we convert the image to grayscale before
            gray = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2GRAY)
            # Then, cv2.Canny applies Canny filter between low and high threshold and returns a binary image with binary matrix of 255 for edges
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            # As usual, we convert this single-channel image back to a three-channel image
            self._current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.add_to_history()


    def adjust_brightness(self, value: int) -> None:
        """Adjust brightness of the image

        Args:
            value: Brightness value, int

        Brightness is just the intensity of the pixel. The range of brightness can be between -100 to 100. A value of -100 subtracts 100 from pixel, making it dark. A value of 0 is default, and a value of 100 adds 100 to the brighness, making it light. However, all values need to be clamped to (0, 255) in the end since brightness can only be between 0 and 255. We do not implement this here, we do this during save.
        """
        if self._current_image is not None:
            # Adding brightness value directly to the image pixel could cause overflow since image is uint8 from 0-255. Adding value = 100 could cause 255 to wrap back to 44.
            # Hence, we convert it to int16 first, since it can hold 37k values
            # Then, we clamp the value back to uint8 (0-255), just as cv2 intended
            self._current_image = np.clip(self._current_image.astype(np.int16) + value, 0, 255).astype(np.uint8)
            self.add_to_history()

    def adjust_contrast(self, value: float) -> None:
        """Adjust contrast of the image

        Args:
            value: Contrast change, float

        Contrast is the difference in brightness between adjacent areas of an image. High contrast = big difference between light and dark areas, low contrast = small difference. The range we have here is 0.1 to 3.0. 0.1 = very washed out, 3.0 = very contrasty. The value is multiplied, so value >1 makes it contrasty, while value <1 makes it less contrasty.
        """
        # Tried implementing the contrast function from the same brightness function above:
        # np.clip(self._current_image.astype(np.int16) * value, 0, 255).astype(np.uint8)
        # BUT this increased both brightness and contrast. The right way, when changing contrast, has to preserve the midpoint values while compressing the values on the right and left tail of the median 128
        # So this function has to be implemented differently. Directly from the docs:
        if self._current_image is not None:
            self._current_image = cv2.convertScaleAbs(self._current_image, alpha=value, beta=0)
            self.add_to_history()

    # SET 3 - IMAGE TRANSFORMATION FUNCTIONS. TBD: Aryan.

    def rotate_image(self, angle: int) -> None:
        """Rotate the image by given angle

        Args:
            angle: int. This will be predefined, so the users can't just rotate it to an arbitrary angle (rarely required).
        """
        if self._current_image is not None:
            source = self._current_image
        else:
            source = self._original_image

        if source is None:
            return

        if angle == 90:
            rotated = rotate(source, ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = rotate(source, ROTATE_180)
        elif angle == 270:
            rotated = rotate(source, ROTATE_90_COUNTERCLOCKWISE)
        else:
            return

        self._current_image = rotated
        self.add_to_history()
        print(f"This function rotates the image at {self._filename}")
        
    def flip_image(self, direction: str) -> None:
        """Flip image, horizontally or vertically along the X and Y axis

        Args:
            direction: "horizontal" | "vertical"

        """
        if self._current_image is not None:
            source = self._current_image
        else:
            source = self._original_image
        if source is None:
            return

        if direction == "horizontal":
            result = flip(source, 1)
        elif direction == "vertical":
            result = flip(source, 0)
        else:
            return

        self._current_image = result
        self.add_to_history()
        print("This function flips image.")


    def resize_image(self, width: int, height: int) -> None:
        """Resize image to a given dimension

        Args:
            width: new width of the image
            height: new height of the image
        """
        if self._current_image is not None:
            source = self._current_image
        else:
            source = self._original_image
        if source is None:
            return

        if width <= 0 or height <= 0:
            return

        self._current_image = resize(source, (width, height), interpolation=INTER_LINEAR)
        self.add_to_history()
        print("This function resizes the image")


    def get_current_image(self) -> Optional[np.ndarray]:
        """Get current image as numpy array"""
        image = self._current_image if self._current_image is not None else self._original_image
        print("Returning image as numpy array")
        return image

    def can_undo(self) -> bool:
        """Is undo operation possible? Only if the stack has some data can undo be done"""
        print("Checking can undo...")
        return self._history_index > 0


    def can_redo(self) -> bool:
        """Check if redo operation is possible. If stack isn't full, can be done"""
        print("Checking can redo...")
        return self._history_index < len(self._history) - 1

    def undo(self) -> bool:
        """Undo last operation."""
        if not self.can_undo():
            print("Undo")
            return False

        self._history_index -= 1
        self._current_image = self._history[self._history_index].copy()
        print("Undo")
        return True


    def redo(self) -> bool:
        """Redo last undone operation."""
        if not self.can_redo():
            print("Redo")
            return False

        self._history_index += 1
        self._current_image = self._history[self._history_index].copy()
        print("Redo")
        return True

    def add_to_history(self) -> None:
        """Add current image state to history stack"""
        if self._current_image is None:
            return

        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]

        self._history.append(self._current_image.copy())
        if len(self._history) > self._max_history:
            self._history.pop(0)

        self._history_index = len(self._history) - 1
        print("Adding image to history stack")

    def clear_history(self) -> None:
        """Clear all operation history. Destroy everything in the history stack."""
        self._history = []
        self._history_index = -1
        print("Clearing history.")

    # SET 4: Additional functions, because the preview implementation needs special functions to not apply cumulative values and pollute the history stack with incorrect values. TBD: Yasmeen

    def start_preview(self) -> None:
        """Save current state before slider adjustment begins"""
        # very important!!!! you will DEFINITELY mess up things if previews are applied on current image and NOT a copy of the current image
        print("Copying current image to preview base")
        if self._current_image is None:
            return
        self._preview_base = self._current_image.copy()

    def cancel_preview(self) -> None:
        """Revert to state before preview started. Just revert what we did above."""
        print("Copying preview base to current image")
        if self._preview_base is None:
            return
        self._current_image = self._preview_base.copy()
        self._preview_base = None

    def commit_preview(self, clear_base: bool = True) -> None:
        """Finalize preview, add to history. Only when a user is satisfied with the preview"""
        print("Adding current image to history")
        if self._current_image is None:
            return
        self._history.append(self._current_image.copy())

        if clear_base:
            self._preview_base = None

    def apply_blur_preview(self, intensity: int) -> None:
        """Apply blur for real-time preview (no history)"""
        print("Applying blur to preview base, and setting it as current image")
        if self._preview_base is None:
            return
        k = max(1, intensity * 2 + 1)
        blurred = cv2.GaussianBlur(self._preview_base, (k, k), 0)
        self._current_image = blurred

    def adjust_brightness_preview(self, value: int) -> None:
        """Adjust brightness for real-time preview (no history)"""
        print("Applying brightness to preview base, and setting it as current image")
        if self._preview_base is None:
            return
        bright = cv2.convertScaleAbs(
            self._preview_base,
            alpha=1.0,
            beta=value
        )
        self._current_image = bright

    def adjust_contrast_preview(self, value: float) -> None:
        """Adjust contrast for real-time preview (no history)"""
        print("Applying contrast to preview base, and setting it as current image")
        if self._preview_base is None:
            return
        contrast = cv2.convertScaleAbs(
            self._preview_base,
            alpha=value,
            beta=0
        )
        self._current_image = contrast
