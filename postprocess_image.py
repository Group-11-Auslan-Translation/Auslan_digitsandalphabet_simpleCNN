import cv2
import numpy as np

def remove_black_borders(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found, get the largest one and crop the image
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_image = image[y:y+h, x:x+w]
        cv2.imwrite(output_path, cropped_image)
    else:
        # If no contours found, save the original image
        cv2.imwrite(output_path, image)

# Example usage:
image_path = "preprocessed_augmented_data/0_augmented_0.png"
output_path = "cropped_image_no_borders.png"

remove_black_borders(image_path, output_path)

print("The image has been processed and saved without borders.")

