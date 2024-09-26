import torchvision.transforms as transforms
from PIL import Image
import os

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.RandomRotation(15),           # Randomly rotate images by ±15 degrees
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # Randomly crop and resize to 256x256
    transforms.RandomHorizontalFlip(),       # Randomly flip images horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Random color changes
    transforms.RandomAffine(15),             # Random affine transformations
    transforms.ToTensor()                    # Convert the PIL image to a tensor
])

# Assuming you have a function to save tensors back as images
def save_tensor_as_image(tensor, output_path):
    # Convert the tensor back to a PIL image
    image = transforms.ToPILImage()(tensor).convert("L")
    image.save(output_path)

# Paths to your original images
input_folder = "preprocessed_data"
output_folder = "augmented_data"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Apply the transformations and save augmented images
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert("L")  # Convert to grayscale

        # Generate multiple augmented images from the same source
        for i in range(5):  # Generate 5 new images per original image
            augmented_image = transform(image)
            output_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_augmented_{i}.png")
            save_tensor_as_image(augmented_image, output_image_path)

print("Augmented images have been generated and saved.")
