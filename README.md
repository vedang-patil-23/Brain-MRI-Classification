# Brain-Tumor-Classification-Using-MRI

## Overview
This repository contains Jupyter Notebook code for training and evaluating various machine learning models to classify brain MRI images. The notebook demonstrates the use of Convolutional Neural Networks (CNNs) and Transfer Learning with popular architectures like VGG16, ResNet50, DenseNet121, EfficientNetB0, InceptionV3, and Xception. Additionally, it includes traditional machine learning models like Support Vector Machines (SVM) and Random Forest Classifiers for comparison.

## Contents
1. **Data Preparation**
   - Load and preprocess brain MRI images.
   - Use `ImageDataGenerator` for data augmentation and splitting into training and validation sets.

2. **Model Training and Evaluation**
   - **CNN Model:** A custom CNN architecture is built and trained.
   - **Transfer Learning Models:**
     - **VGG16**
     - **ResNet50**
     - **DenseNet121**
     - **EfficientNetB0**
     - **InceptionV3**
     - **Xception**
   - **Traditional Machine Learning Models:**
     - **Support Vector Machine (SVM)**
     - **Random Forest Classifier**

3. **Performance Metrics**
   - Accuracy and loss plots for each model.
   - Confusion matrices to evaluate the performance of each model on the validation set.

4. **Prediction**
   - Functions to predict the class of new MRI images using the trained models.

## Requirements
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

To install the necessary packages, you can use:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## Data
The notebook assumes that the data is organized in the following directory structure:
```
data/
    Train/
        Normal/
        Tumor/
    Validation/
        Normal/
        Tumor/
```
Please update the `train_dir` and `val_dir` variables in the notebook to point to your data directories.

## Code Overview

### 1. Data Preparation
```python
train_dir = "path/to/train"
val_dir = "path/to/validation"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(150, 150),
                                                batch_size=32,
                                                class_mode='binary')
```

### 2. CNN Model
```python
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_history = cnn_model.fit(train_generator, validation_data=val_generator, epochs=10)
```

### 3. Transfer Learning Models
Each transfer learning model follows a similar pattern:
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

vgg_base = VGG16(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
for layer in vgg_base.layers:
    layer.trainable = False

vgg_model = Sequential([
    vgg_base,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
vgg_history = vgg_model.fit(train_generator, validation_data=val_generator, epochs=10)
```

### 4. Traditional Machine Learning Models
SVM and Random Forest models are trained using features extracted from a pre-trained VGG16 model.
```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Feature extraction and model training
```

### 5. Prediction Functions
Functions to predict new images:
```python
def predict_with_cnn(image_path, model):
    # Prediction code
```
Update `image_path` to your image file and call the function with the model you wish to use.

## Usage
1. Modify the paths to your dataset in the `train_dir` and `val_dir` variables.
2. Run the notebook to train the models and evaluate their performance.
3. Use the provided prediction functions to classify new MRI images.

## Results
The notebook includes plots for accuracy and loss curves, as well as confusion matrices for each model.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or issues, please contact [Me](mailto:vedangrpatil.2305@gmail.com).

