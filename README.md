# Brain MRI Classification

This repository contains Jupyter Notebook code for classifying brain MRI images using various deep learning models. The code covers multiple approaches, including Convolutional Neural Networks (CNNs), transfer learning with pre-trained models, and feature extraction for traditional machine learning classifiers.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Models](#models)
  - [CNN Model](#cnn-model)
  - [VGG16 Model](#vgg16-model)
  - [ResNet50 Model](#resnet50-model)
  - [DenseNet121 Model](#densenet121-model)
  - [EfficientNetB0 Model](#efficientnetb0-model)
  - [InceptionV3 Model](#inceptionv3-model)
  - [Xception Model](#xception-model)
- [Feature Extraction and Classification](#feature-extraction-and-classification)
  - [SVM Classifier](#svm-classifier)
  - [Random Forest Classifier](#random-forest-classifier)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [License](#license)

## Overview

This notebook demonstrates several approaches to classifying brain MRI images. The methods include:

- Custom CNN architectures
- Transfer learning with popular pre-trained models
- Feature extraction and classification with traditional machine learning classifiers
- Dataset was used from Kaggle Dataset Repositories [Link to Dataset](https://www.kaggle.com/datasets/muhammadehsan000/brain-mri-imaging-data)

## Dependencies

The following libraries are required:

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

You can install the dependencies using pip:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## Data Preparation

The dataset consists of brain MRI images divided into training and validation sets. The images are stored in the following directories:

- `Train`: Contains training images
- `Validation`: Contains validation images

The images are preprocessed using `ImageDataGenerator` to perform data augmentation and normalization.

## Models

### CNN Model

A custom Convolutional Neural Network (CNN) is built and trained for binary classification (Tumor vs. Normal). The CNN consists of multiple convolutional and pooling layers followed by fully connected layers.

### VGG16 Model

Transfer learning is used with the VGG16 model. The pre-trained VGG16 model is fine-tuned for binary classification by adding a few dense layers on top of it.

### ResNet50 Model

Similarly, the ResNet50 model is utilized for transfer learning. The ResNet50 architecture is adapted for binary classification tasks.

### DenseNet121 Model

The DenseNet121 model is employed using transfer learning to classify brain MRI images. The pre-trained DenseNet121 is fine-tuned for the binary classification task.

### EfficientNetB0 Model

EfficientNetB0, another state-of-the-art pre-trained model, is utilized for brain MRI classification using transfer learning.

### InceptionV3 Model

InceptionV3 is used as a pre-trained model for feature extraction and classification of brain MRI images.

### Xception Model

The Xception model is employed for classification tasks by leveraging transfer learning and fine-tuning the pre-trained model.

## Feature Extraction and Classification

### SVM Classifier

Features are extracted from the VGG16 model and used to train a Support Vector Machine (SVM) classifier for brain MRI classification.

### Random Forest Classifier

Similarly, features extracted from the VGG16 model are used to train a Random Forest classifier.

## Evaluation

The performance of each model is evaluated using accuracy, confusion matrices, and plots showing accuracy and loss curves over epochs. 

## Usage

To use the trained models for prediction, you can call the provided functions with the path to an MRI image. Example:

```python
from tensorflow.keras.preprocessing import image

def predict_with_cnn(image_path, model):
    img = image.load_img(image_path, target_size=(150, 150))  # Adjust target_size if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Preprocess for CNN model

    prediction = model.predict(img_array)[0][0]
    label = 'Brain with Tumor' if prediction > 0.5 else 'Normal Brain'

    plt.imshow(img)
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

    print(f"The selected Brain MRI Image is of a {label}")

# Example usage
image_path = "path/to/your/image.jpg"
predict_with_cnn(image_path, cnn_model)
```

Replace `"path/to/your/image.jpg"` with the path to your image file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

