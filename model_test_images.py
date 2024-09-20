import argparse
import os
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from main import HandSignClassifier


def preprocess_test_image(image_path, target_size=(256, 256)):
    """
    Preprocesses the image: resize, pad, apply Gaussian blurring, and adaptive thresholding.
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

    # Create a new image with the target size and a white background color
    background_color = 255  # White background for grayscale
    new_image = Image.new("L", target_size, background_color)
    new_image.paste(resized_image, ((target_size[0] - resized_image.size[0]) // 2,
                                    (target_size[1] - resized_image.size[1]) // 2))

    # Convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    tensor = transform(new_image).unsqueeze(0)  # Add batch dimension
    return tensor.float()


def test_model_on_directory(image_dir, model_path):
    # Load the trained model
    model = HandSignClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Iterate over all images in the directory
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Preprocess the image
            unseen_image = preprocess_test_image(image_path)
            unseen_image = unseen_image.to(device)

            # Perform inference
            with torch.no_grad():
                outputs = model(unseen_image)
                _, predicted_label = torch.max(outputs, 1)

            # Print the predicted label
            print(f"Image: {image_file} -> Predicted Label: {predicted_label.item()}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test a trained model on all images in a directory.")
    parser.add_argument("image_dir", type=str, help="Directory containing test images")
    parser.add_argument("--model_path", type=str, default="number_recognition_model.pth",
                        help="Path to the trained model")

    args = parser.parse_args()

    # Run the test on all images in the directory
    test_model_on_directory(args.image_dir, args.model_path)
