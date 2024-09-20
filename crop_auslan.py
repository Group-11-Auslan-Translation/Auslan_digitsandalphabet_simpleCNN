from PIL import Image, ImageOps, ImageFilter
import os
import cv2
import numpy as np

def resize_and_pad_image(image_path, output_path, target_size=(256, 256), background_color=255):
    """
    Resize the image to fit within the target size while maintaining the aspect ratio.
    The image is then padded to fill the target size.
    """
    image = Image.open(image_path)

    # Apply Gaussian blurring
    image = image.filter(ImageFilter.GaussianBlur(radius=2))

    # Convert to grayscale
    image = image.convert("L")

    # Convert the image to a NumPy array for further processing
    image_np = np.array(image)

    # Apply adaptive thresholding
    image_np = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Convert back to a PIL image
    image = Image.fromarray(image_np)

    # Resize the image while maintaining the aspect ratio
    resized_image = ImageOps.contain(image, target_size)

    # Create a new image with the target size and the background color
    new_image = Image.new("L", target_size, background_color)  # "L" for grayscale

    # Paste the resized image onto the center of the new image
    new_image.paste(resized_image, ((target_size[0] - resized_image.size[0]) // 2,
                                    (target_size[1] - resized_image.size[1]) // 2))

    # Save the processed image
    new_image.save(output_path)


# Paths to your cropped images
input_folder = "data/auslan_images"
output_folder = "resized_data"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each image (assuming they are named as 0.png, 1.png, ..., 9.png)
for i in range(10):
    input_image_path = os.path.join(input_folder, f"{i}.png")
    output_image_path = os.path.join(output_folder, f"{i}.png")

    resize_and_pad_image(input_image_path, output_image_path)

print("Images have been resized, padded, preprocessed, and saved successfully.")
