# Auslan_digitandalphabets_simpleCNN

# Hand Sign (AUSLAN) Number (0-9) and Alphabets (A-Z) Recognition with CNN

This project implements a simple Convolutional Neural Network (CNN) to classify hand sign images representing numbers (0-9) and Alphabets (A-Z). The model is trained on grayscale images that have been preprocessed using techniques like resizing, padding, Gaussian blurring, and adaptive thresholding.

## Project Structure

## Model Architecture

The model is a simple CNN with the following layers:
- **Input Layer**: Grayscale images with size 256x256 pixels.

- **4 Convolutional Layers**:

conv1: 1 input channel, 32 output channels, 3x3 kernel, ReLU activation.

conv2: 32 input channels, 64 output channels, 3x3 kernel, ReLU activation.

conv3: 64 input channels, 128 output channels, 3x3 kernel, ReLU activation.

conv4: 128 input channels, 256 output channels, 3x3 kernel, ReLU activation.

3 Max Pooling Layers: 2x2 kernel with stride 2.

- **2 Fully Connected Layers**:

fc1: 1024 neurons, ReLU activation.
fc2: 36 output neurons (one for each digit 0-9 and letter A-Z).
## Preprocessing

Images are taken from https://auslan.org.au/numbersigns.html and https://auslan.org.au/spell/twohanded.html Signbank, cropped in an image editor.

Before training, images are preprocessed using the following steps:
1. **Resizing and Padding**: Images are resized to fit within a 256x256 pixel square while maintaining aspect ratio. They are then padded with a white background.
2. **Gaussian Blurring**: Applied to reduce noise and smooth the images.
3. **Adaptive Thresholding**: Converts images to binary format, emphasizing the hand signs.

These steps are implemented in `crop_auslan.py`.

## Training

To train the model, run the `main.py` script on images located in 'preprocessed_data':

python main.py 

The model will be saved as the 'number_recognition_model.pth' in the main directory.

## Testing

To run the test script in model_test_images.py for testing on all test images in a directory, use the following command:

python model_test_images.py /path/to/test_images_directory --model_path /path/to/your/model.pth

- image_dir: This is a required argument that specifies the directory containing the test images.
- model_path: This is an optional argument that allows you to specify a custom path to your trained model. If not provided, it defaults to "number_recognition_model.pth"

Or for the current project scenario:

python model_test_images.py test_data --model_path number_recognition_model.pth for this project.
"# Auslan_0-9digit_simpleCNN" 
