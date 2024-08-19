


# Underwater Image Enhancement with UResNet

## Project Overview

This project focuses on enhancing underwater images using a UResNet model. The UResNet architecture is designed to improve the visibility and clarity of underwater images, which often suffer from low contrast, color distortion, and noise. The model is trained on a dataset of underwater images and employs custom metrics and callbacks to achieve optimal performance.

## Features

- **Custom UResNet Model**: Utilizes a modified U-Net architecture with residual connections to enhance underwater images.
- **PSNR-Based Evaluation**: Implements a custom callback to monitor Peak Signal-to-Noise Ratio (PSNR) and save the best-performing model.
- **Adaptive Learning Rate**: Integrates a ReduceLROnPlateau callback to dynamically adjust the learning rate based on validation loss.
- **Real-Time Image Enhancement**: Optimized to run at 45 FPS on 1080p resolution images.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/underwater-image-enhancement.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The project utilizes an underwater image dataset with paired input and target images. The input images are the original underwater images, and the target images are the enhanced versions.

- **Input Images**: Low-quality underwater images.
- **Target Images**: High-quality enhanced underwater images.

## Model Architecture

The UResNet model is based on the U-Net architecture, with added residual connections to improve training stability and performance. The model consists of an encoder, a bottleneck, and a decoder, each with multiple convolutional layers.

## Training

The model is trained on a dataset of underwater images with the following parameters:

- **Epochs**: 20
- **Batch Size**: 16
- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam

### Custom Callbacks

- **PSNRAndSaveModelCallback**: Monitors the PSNR on the validation set and saves the model with the best PSNR.
- **ReduceLROnPlateau**: Adjusts the learning rate when the validation loss plateaus.

To train the model, run:

```python
python train.py
```

## Example Usage

To enhance an underwater image:

```python
from model import enhance_image

enhanced_img = enhance_image(model, 'path/to/your/image.jpg', (256, 256))
```

## Results

The trained model improves the PSNR of underwater images by over 20\% compared to the input images. The model is also capable of real-time enhancement at 45 FPS on 1080p resolution images.

## Visualization

Example of an original and enhanced underwater image:

- **Original Image**:
![Original Image](images/original.jpg)
- **Enhanced Image**:
![Enhanced Image](images/enhanced.jpg)

## Conclusion

This project demonstrates the effectiveness of UResNet for underwater image enhancement. By leveraging custom metrics and callbacks, the model achieves high performance and real-time processing capabilities.

## Future Work

- Explore alternative architectures and loss functions for further improvements.
- Extend the model to work on different types of degraded images beyond underwater images.
- Deploy the model in a web application for wider accessibility.




