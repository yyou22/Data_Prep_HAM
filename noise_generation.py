import cv2
import numpy as np
import os

def normalize_image(img, min_val, max_val):
    return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def main():
    folder1 = "Images"
    folder2 = "Images2"
    output_folder = "noise_2"

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in folder1
    filenames = os.listdir(folder1)
    
    min_all = float("inf")
    max_all = float("-inf")

    # Store all subtracted images here first
    subtracted_images = []

    for filename in filenames:
        # Construct full path for both folders
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)

        # Read the images
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None or img2 is None:
            print(f"Could not read one of the images: {path1} or {path2}")
            continue

        # Subtract the images
        subtracted = np.subtract(img1.astype(np.float32), img2.astype(np.float32))

        # Update min and max values
        min_all = min(min_all, np.min(subtracted))
        max_all = max(max_all, np.max(subtracted))

        subtracted_images.append((filename, subtracted))

    # Normalize and save all images based on min_all and max_all
    for filename, img in subtracted_images:
        normalized_img = normalize_image(img, min_all, max_all)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, normalized_img)

if __name__ == "__main__":
    main()
