# CIFAR10-TensorFlow-CNN: CIFAR-10 Image Classification with Convolutional Neural Networks

This repository contains a Python-based implementation of CIFAR-10 image classification using Convolutional Neural Networks (CNNs) developed with TensorFlow and Keras. The project explores model optimization techniques, including data augmentation and batch normalization, to improve performance.

## Features

- **CIFAR-10 Dataset**: Preprocessed the dataset with normalization and splitting into training, validation, and testing sets.
- **Model Architectures**: Developed CNN models with varying configurations, including options for data augmentation and batch normalization.
- **Optimization Techniques**: Utilized Adam optimizer and evaluated learning rates to refine training.
- **Performance Evaluation**: Visualized training and validation loss trends and tested the model on unseen data.
- **Data Augmentation**: Implemented augmentation techniques such as rotation, width/height shifts, and horizontal flips to enhance generalization.

## Requirements

- Python 3.7+
- TensorFlow 2.0+
- NumPy
- Matplotlib
- Scikit-learn

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 50,000 for training and 10,000 for testing. The dataset is automatically downloaded when the script is run.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/sirnerd030/CIFAR10-TensorFlow-CNN.git
   cd CIFAR10-TensorFlow-CNN
   ```

2. Run the training script:
   ```bash
   python cifar10_training.py
   ```

3. The script will output training and validation loss/accuracy and save the best-performing models.

## Project Structure
```
.
├── cifar10_training.py     # Main script for training and evaluating models
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── models/                 # Directory for saving trained models
```

## Results
- The best-performing model achieved over 80% accuracy on the CIFAR-10 test set.
- Visualizations of training and validation loss trends demonstrated model improvements with data augmentation and batch normalization.

## Future Work
- Explore more advanced architectures, such as ResNet or VGG.
- Implement transfer learning using pre-trained models.
- Optimize hyperparameters with tools like Grid Search or Bayesian Optimization.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- TensorFlow and Keras documentation
- CIFAR-10 dataset

## Contact
For any questions or suggestions, feel free to reach out at [your email address] or open an issue on this repository.
