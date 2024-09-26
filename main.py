import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import glob
import os

# Custom Dataset Class for Number Signs
class HandSignDataset(torch.utils.data.Dataset):
    """
    A class which loads images of hand signs representing numbers (0-9) and letters (A-Z),
    including cases where letters have sequences (e.g., J_1, J_2, J_3).
    """
    def __init__(self, images_path="data/preprocessed_data", dim=256):
        super(HandSignDataset, self).__init__()

        # Set image paths and resize dimension
        self.images_path = images_path
        self.dim = dim

        # Load image paths and their corresponding labels
        self.image_paths = glob.glob(os.path.join(self.images_path, '*.png'))

        # Label extraction logic
        self.labels = []
        for path in self.image_paths:
            filename = os.path.basename(path).split('.')[0]  # Extract the name before the file extension
            label_part = filename.split('_')[0]  # Extract the main label part before the first '_'

            if label_part.isdigit():
                label = int(label_part)  # Convert numbers (0-9) directly to integers
            elif len(label_part) == 1 and label_part.isalpha():
                label = ord(label_part.upper()) - ord('A') + 10  # Convert letters (A-Z) to 10-35
            else:
                raise ValueError(f"Unexpected label format in filename: {filename}")

            self.labels.append(label)

    def process_image(self, data_path):
        image = Image.open(data_path)
        resized = image.resize((self.dim, self.dim), resample=Image.NEAREST)
        tensor = torchvision.transforms.functional.pil_to_tensor(resized) / 255.0  # Normalize to [0,1]
        return tensor.float()

    def __getitem__(self, index):
        data_path = self.image_paths[index]
        data = self.process_image(data_path)

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return data, label

    def __len__(self):
        return len(self.image_paths)

# Updated HandSignClassifier with an extra convolutional layer
class HandSignClassifier(nn.Module):
    def __init__(self):
        super(HandSignClassifier, self).__init__()

        # Updated CNN architecture with an extra convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)    # 1 input channel (grayscale), 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 32 input channels, 64 output channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 input channels, 128 output channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # 128 input channels, 256 output channels

        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by a factor of 2
        self.relu = nn.ReLU()

        # Adjusted fully connected layers for classification
        self.fc1 = nn.Linear(256 * 16 * 16, 1024)  # Adjusted input size after extra convolutional layers and pooling
        self.fc2 = nn.Linear(1024, 36)  # 36 output classes (0-9 and A-Z)

    def forward(self, x):
        # Forward pass through CNN layers with pooling
        x = self.pool(self.relu(self.conv1(x)))  # After conv1 and pooling: [batch_size, 32, 128, 128]
        x = self.pool(self.relu(self.conv2(x)))  # After conv2 and pooling: [batch_size, 64, 64, 64]
        x = self.pool(self.relu(self.conv3(x)))  # After conv3 and pooling: [batch_size, 128, 32, 32]
        x = self.pool(self.relu(self.conv4(x)))  # After conv4 and pooling: [batch_size, 256, 16, 16]

        # Flatten the feature map for fully connected layers
        x = x.view(-1, 256 * 16 * 16)

        # Fully connected layers
        x = self.relu(self.fc1(x))  # First fully connected layer with ReLU activation
        x = self.fc2(x)             # Output layer with logits for 10 classes

        return x
# Loss Function
def loss_function(model_output, targets):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model_output, targets)
    return loss

# Training function
def train_model():
    # Define training parameters
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001

    # Create the dataset and dataloader
    dataset = HandSignDataset(images_path="preprocessed_data", dim=256)  # Ensure this path is correct
    print(f"Number of images found: {len(dataset)}")

    # Check if images are loaded correctly
    if len(dataset) == 0:
        raise ValueError("No images found. Please check the file paths and ensure images are available.")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = HandSignClassifier()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Check if CUDA is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), "number_recognition_model.pth")

if __name__ == "__main__":
    train_model()
