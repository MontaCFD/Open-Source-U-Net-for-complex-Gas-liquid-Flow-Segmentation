import os
import numpy as np
import cv2
import matplotlib as plt
import glob

script_dir = r"C:\Users\Niepraschk\Desktop\Fuer Manuel\Diplomarbeit\Austausch\von Manuel\ZET\Aufnahmen45grad"
images_dir = os.path.join(script_dir, "26_03_25-020_2135")  # Update as needed

background_file = "background.bmp"
image_pattern = "Bild*.bmp"

def load_images():
    """Loads the background and all experiment frames."""
    background_path = os.path.join(images_dir, background_file)
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise FileNotFoundError(f"Could not load background image: {background_path}")
    all_files = sorted(glob.glob(os.path.join(images_dir, image_pattern)))
    if len(all_files) < 2:
        raise FileNotFoundError("Need at least two images to track velocity.")
    return background, all_files

roi_offset = 50
base_tol = 10    # Base tolerance in intensity difference
rel_tol = 0.1    # Relative tolerance factor (10% of background pixel value)

def adaptive_background_subtraction(frame_gray, background, base_tol, rel_tol):
    """
    Computes the absolute difference between frame and background.
    For each pixel, a dynamic tolerance is computed as:
         tol_pixel = base_tol + rel_tol * background_pixel_value.
    Pixels with a difference below this tolerance are set to 0.
    """
    diff = cv2.absdiff(frame_gray, background).astype(np.float32)
    dynamic_tol = base_tol + rel_tol * background.astype(np.float32)
    diff[diff < dynamic_tol] = 0
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return diff

def apply_roi(diff):
    """
    Zero out the left & right columns in 'diff' to ignore shadows.
    """
    h, w = diff.shape
    diff[:, :roi_offset] = 0
    diff[:, w - roi_offset :] = 0
    return diff

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(12,12)):
    """
    Enhances local contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters:
      - image: The input grayscale image.
      - clip_limit: Threshold for contrast limiting. Lower values produce a more subtle effect,
                    while higher values allow greater contrast but can amplify noise.
      - tile_grid_size: Size (in number of tiles) for the grid used in CLAHE. Smaller grid sizes
                        (e.g., (4,4)) enhance very local details; larger grid sizes (e.g., (16,16))
                        provide more global contrast enhancement.

    Returns:
      - The CLAHE-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

if __name__ == "__main__":
    background, all_files = load_images()

    output_folder = os.path.join(script_dir, "Backgroundremoval2")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_path in all_files:
        # Lade das aktuelle Bild
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Fehler: Bild konnte nicht geladen werden: {file_path}")
            continue

        # Verarbeitungsschritte
        diff_image = adaptive_background_subtraction(image, background, base_tol, rel_tol)
        crop_image = apply_roi(diff_image)
        new_image = apply_clahe(crop_image)

        # Speichern des verarbeiteten Bildes als .tif
        file_name = os.path.splitext(os.path.basename(file_path))[0] + ".tif"
        output_path = os.path.join(output_folder, file_name)

        # Speichern mit OpenCV
        cv2.imwrite(output_path, new_image)

        print(f"Bild erfolgreich gespeichert als TIFF unter: {output_path}")
