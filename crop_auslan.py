import cv2
import os
from PIL import Image


def apply_gaussian_blur_and_threshold(image_path, output_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply adaptive thresholding
    threshold_image = cv2.adaptiveThreshold(blurred_image, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    # Save the processed image
    cv2.imwrite(output_path, threshold_image)


# Paths to your augmented images
input_folder = "auslan_images"
output_folder = "preprocessed_data"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each augmented image
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename)
        apply_gaussian_blur_and_threshold(input_image_path, output_image_path)

print("Gaussian blur and thresholding applied to augmented images and saved successfully.")
